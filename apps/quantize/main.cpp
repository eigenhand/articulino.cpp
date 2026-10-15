// articulino-quantize: requantize the matmul weights of an F32 (or F16) articulino GGUF with ggml's own quantizers,
// including the K-quants, which the Python converter cannot write. 4-bit types are not offered: in our evaluation they
// sounded measurably worse than F32 (see MODEL_CARD_GGUF.md); 8 bits is the smallest released precision.
//
//   articulino-quantize in-f32.gguf out.gguf TYPE [--voice TYPE] [--enc TYPE] [--keep TYPE:REGEX]...
//
// TYPE: f16 q8_0 q6_k q5_k q5_0 (q6_k/q5_k/q5_0 are not evaluated). Matmul weights are 2-D tensors named *.weight with at least 32 rows;
// everything else (norms, biases, embeddings, depthwise taps, mel filters) is copied unchanged. A weight whose row
// length does not fit the block size falls back to the next type in: TYPE -> q5_0/q8_0 -> f16.
// --voice sets the type of the voice encoder (voice.*, default f16) and --enc that of the latent encoder (enc.*, default
// q8_0 for 5-bit types); both run once per voice, not per sentence.
// --keep TYPE:REGEX overrides the type for matching tensor names (first match wins; later --keep beats the defaults).
// Defaults: enc.mu stays F16 in every file (its inputs have outliers); in 5-bit files dit.out and dec.head stay q8_0.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

static ggml_type parse_type(const std::string & s) {
    static const std::map<std::string, ggml_type> m = {
        { "f32", GGML_TYPE_F32 }, { "f16", GGML_TYPE_F16 }, { "q8_0", GGML_TYPE_Q8_0 }, { "q6_k", GGML_TYPE_Q6_K },
        { "q5_k", GGML_TYPE_Q5_K }, { "q5_0", GGML_TYPE_Q5_0 },
    };
    auto it = m.find(s);
    if (it == m.end()) { std::fprintf(stderr, "unknown type %s\n", s.c_str()); std::exit(2); }
    return it->second;
}

static ggml_type fallback(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q5_K: return GGML_TYPE_Q5_0;
        case GGML_TYPE_Q6_K: return GGML_TYPE_Q8_0;
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0: return GGML_TYPE_F16;
        default: return GGML_TYPE_F16;
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: articulino-quantize in.gguf out.gguf TYPE [--voice TYPE] [--keep TYPE:REGEX]... [--name NAME]\n");
        return 2;
    }
    const std::string in = argv[1], out = argv[2], tname = argv[3];
    const ggml_type base = parse_type(tname);
    ggml_type voice = GGML_TYPE_F16, enc = base;
    bool enc_set = false;
    std::string label = tname;
    std::vector<std::pair<std::regex, ggml_type>> keep;
    std::vector<std::pair<std::regex, ggml_type>> user_keep;
    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--voice" && i + 1 < argc) voice = parse_type(argv[++i]);
        else if (a == "--enc" && i + 1 < argc) { enc = parse_type(argv[++i]); enc_set = true; }
        else if (a == "--keep" && i + 1 < argc) {
            const std::string k = argv[++i];
            const size_t c = k.find(':');
            if (c == std::string::npos) { std::fprintf(stderr, "--keep needs TYPE:REGEX\n"); return 2; }
            user_keep.emplace_back(std::regex(k.substr(c + 1)), parse_type(k.substr(0, c)));
        } else if (a == "--name" && i + 1 < argc) label = argv[++i];
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    // later --keep options first, then the defaults for low-bit files
    for (auto it = user_keep.rbegin(); it != user_keep.rend(); ++it) keep.push_back(*it);
    const bool low_bit = base != GGML_TYPE_F32 && base != GGML_TYPE_F16 && base != GGML_TYPE_Q8_0 && base != GGML_TYPE_Q6_K;
    if (low_bit && !enc_set) enc = GGML_TYPE_Q8_0;
    if (base == GGML_TYPE_F32 || base == GGML_TYPE_F16) voice = base;
    keep.emplace_back(std::regex("^enc\\.mu\\."), GGML_TYPE_F16);   // outlier inputs: keep 16 bits in every file
    if (low_bit) keep.emplace_back(std::regex("^(dit\\.out\\.|dec\\.head)"), GGML_TYPE_Q8_0);

    ggml_context * src_ctx = nullptr;
    gguf_init_params ip{ false, &src_ctx };
    gguf_context * src = gguf_init_from_file(in.c_str(), ip);
    if (!src) { std::fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_str(dst, "articulino.weight_type", label.c_str());

    const int64_t n = gguf_get_n_tensors(src);
    ggml_init_params cp{ ggml_tensor_overhead() * (n + 1), nullptr, true };
    ggml_context * dst_ctx = ggml_init(cp);
    std::vector<std::vector<uint8_t>> store(n);
    std::map<std::string, double> by_type;
    size_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(src, i);
        ggml_tensor * t = ggml_get_tensor(src_ctx, name);
        const std::string nm = name;
        const bool is_mat = ggml_n_dims(t) == 2 && t->ne[1] >= 32 && nm.size() > 7 && nm.compare(nm.size() - 7, 7, ".weight") == 0 &&
                            (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
        ggml_type want = t->type;
        if (is_mat) {
            want = nm.rfind("voice.", 0) == 0 ? voice : (nm.rfind("enc.", 0) == 0 ? enc : base);
            for (auto & k : keep) if (std::regex_search(nm, k.first)) { want = k.second; break; }
            while (want != GGML_TYPE_F16 && want != GGML_TYPE_F32 && t->ne[0] % ggml_blck_size(want) != 0) want = fallback(want);
        }
        ggml_tensor * o = ggml_new_tensor(dst_ctx, want, ggml_n_dims(t), t->ne);
        ggml_set_name(o, name);
        const int64_t nel = ggml_nelements(t);
        std::vector<float> f32(nel);
        if (t->type == GGML_TYPE_F32) std::memcpy(f32.data(), t->data, nel * sizeof(float));
        else if (t->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f32.data(), nel);
        if (want == t->type) {
            store[i].assign((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));
        } else if (want == GGML_TYPE_F32) {
            store[i].resize(nel * sizeof(float));
            std::memcpy(store[i].data(), f32.data(), store[i].size());
        } else {
            store[i].resize(ggml_nbytes(o));
            const size_t sz = ggml_quantize_chunk(want, f32.data(), store[i].data(), 0, t->ne[1], t->ne[0], nullptr);
            store[i].resize(sz);
        }
        o->data = store[i].data();
        gguf_add_tensor(dst, o);
        by_type[ggml_type_name(want)] += (double) nel;
        total += store[i].size();
    }
    if (!gguf_write_to_file(dst, out.c_str(), false)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::fprintf(stderr, "%s: %.1f MB of tensors |", out.c_str(), total / 1e6);
    for (auto & kv : by_type) std::fprintf(stderr, " %s %.1f M", kv.first.c_str(), kv.second / 1e6);
    std::fprintf(stderr, "\n");
    gguf_free(dst);
    gguf_free(src);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    return 0;
}
