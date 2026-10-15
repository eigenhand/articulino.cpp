// Model loading (GGUF), backend and graph plumbing.
#include "engine.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace articulino {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------ graph

Graph::Graph(size_t n_nodes) {
    ggml_init_params p{ ggml_tensor_overhead() * n_nodes * 2 + ggml_graph_overhead_custom(n_nodes, false) + 65536,
                        nullptr, true };
    ctx = ggml_init(p);
    gf = ggml_new_graph_custom(ctx, n_nodes, false);
}

Graph::~Graph() {
    if (ctx) ggml_free(ctx);
}

ggml_tensor * Graph::in_f32(const float * data, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_input(t);
    const size_t n = (size_t) (ne0 * ne1 * ne2) * sizeof(float);
    std::vector<uint8_t> b(n);
    std::memcpy(b.data(), data, n);
    inputs.emplace_back(t, std::move(b));
    return t;
}

ggml_tensor * Graph::in_i32(const int32_t * data, int64_t ne0) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne0);
    ggml_set_input(t);
    std::vector<uint8_t> b((size_t) ne0 * sizeof(int32_t));
    std::memcpy(b.data(), data, b.size());
    inputs.emplace_back(t, std::move(b));
    return t;
}

void Graph::out(ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
}

void Graph::expand(ggml_tensor * t) {
    ggml_build_forward_expand(gf, t);
}

Voice::~Voice() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
}

// ------------------------------------------------------------------------------------------------ engine

Engine::~Engine() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (galloc_) ggml_gallocr_free(galloc_);
    for (auto b : wbufs_) ggml_backend_buffer_free(b);
    for (auto c : wctxs_) ggml_free(c);
    if (wctx_) ggml_free(wctx_);
    if (gguf_) gguf_free(gguf_);
    if (gpu_) ggml_backend_free(gpu_);
    if (backend_) ggml_backend_free(backend_);
    if (tp_) ggml_threadpool_free(tp_);
}

static int kv_int(gguf_context * g, const std::string & key, int def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32: return (int) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64: return (int) gguf_get_val_i64(g, id);
        default: return def;
    }
}

static float kv_float(gguf_context * g, const std::string & key, float def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(g, id);
    return def;
}

static std::string kv_str(gguf_context * g, const std::string & key, const std::string & def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(g, id);
}

static std::vector<float> kv_floats(gguf_context * g, const std::string & key) {
    std::vector<float> out;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(g, id);
    const gguf_type t = gguf_get_arr_type(g, id);
    const void * d = gguf_get_arr_data(g, id);
    for (size_t i = 0; i < n; i++) {
        if (t == GGUF_TYPE_FLOAT32) out.push_back(((const float *) d)[i]);
        else if (t == GGUF_TYPE_FLOAT64) out.push_back((float) ((const double *) d)[i]);
    }
    return out;
}

static std::vector<std::string> kv_strs(gguf_context * g, const std::string & key) {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) return out;
    const size_t n = gguf_get_arr_n(g, id);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(g, id, i));
    return out;
}

static void log_quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || std::getenv("ARTICULINO_VERBOSE")) std::fputs(text, stderr);
}

bool Engine::load(const std::string & path, int n_threads, bool use_gpu) {
    ggml_log_set(log_quiet, nullptr);
    gguf_init_params gp{ true, &wctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), gp);
    if (!gguf_) { err_ = "cannot read GGUF file: " + path; return false; }
    const std::string arch = kv_str(gguf_, "general.architecture", "");
    if (arch != "articulino") { err_ = "not an articulino model (architecture '" + arch + "')"; return false; }

    const std::string A = "articulino.";
    HParams & h = hp_;
    h.sample_rate = kv_int(gguf_, A + "sample_rate", h.sample_rate);
    h.vocab = kv_int(gguf_, A + "vocab_size", h.vocab);
    h.latent = kv_int(gguf_, A + "latent_dim", h.latent);
    h.samples_per_latent = kv_int(gguf_, A + "samples_per_latent", h.samples_per_latent);
    h.units_per_latent = kv_int(gguf_, A + "duration_units_per_latent", h.units_per_latent);
    h.duration_max = kv_float(gguf_, A + "duration_max", h.duration_max);
    h.prompt_frames = kv_int(gguf_, A + "prompt_frames", h.prompt_frames);
    h.decoder_context = kv_int(gguf_, A + "decoder_context_frames", h.decoder_context);
    h.voice_max_samples = kv_int(gguf_, A + "voice_max_samples", h.voice_max_samples);
    h.text_dim = kv_int(gguf_, A + "text.dim", h.text_dim);
    h.text_layers = kv_int(gguf_, A + "text.layers", h.text_layers);
    h.text_heads = kv_int(gguf_, A + "text.heads", h.text_heads);
    h.voice_dim = kv_int(gguf_, A + "voice.dim", h.voice_dim);
    h.voice_layers = kv_int(gguf_, A + "voice.layers", h.voice_layers);
    h.voice_heads = kv_int(gguf_, A + "voice.heads", h.voice_heads);
    h.voice_tokens = kv_int(gguf_, A + "voice.tokens", h.voice_tokens);
    h.voice_out = kv_int(gguf_, A + "voice.out_dim", h.voice_out);
    h.dur_width = kv_int(gguf_, A + "duration.width", h.dur_width);
    h.dit_dim = kv_int(gguf_, A + "dit.dim", h.dit_dim);
    h.dit_layers = kv_int(gguf_, A + "dit.layers", h.dit_layers);
    h.dit_heads = kv_int(gguf_, A + "dit.heads", h.dit_heads);
    h.rope_base = kv_float(gguf_, A + "rope.base", h.rope_base);
    h.mel_nfft = kv_int(gguf_, A + "mel.n_fft", h.mel_nfft);
    h.mel_win = kv_int(gguf_, A + "mel.win_length", h.mel_win);
    h.mel_hop = kv_int(gguf_, A + "mel.hop", h.mel_hop);
    h.n_mels = kv_int(gguf_, A + "mel.n_mels", h.n_mels);
    h.istft_nfft = kv_int(gguf_, A + "istft.n_fft", h.istft_nfft);
    h.istft_hop = kv_int(gguf_, A + "istft.hop", h.istft_hop);
    h.enc_dim = kv_int(gguf_, A + "encoder.dim", h.enc_dim);
    h.enc_blocks = kv_int(gguf_, A + "encoder.blocks", h.enc_blocks);
    h.dec_dim = kv_int(gguf_, A + "decoder.dim", h.dec_dim);
    h.dec_blocks = kv_int(gguf_, A + "decoder.blocks", h.dec_blocks);
    h.shelf_db = kv_float(gguf_, A + "post.shelf_db", h.shelf_db);
    h.shelf_hz = kv_float(gguf_, A + "post.shelf_hz", h.shelf_hz);
    h.shelf_slope = kv_float(gguf_, A + "post.shelf_slope", h.shelf_slope);
    h.peak = kv_float(gguf_, A + "post.peak", h.peak);
    h.latent_mean = kv_floats(gguf_, A + "latent_mean");
    h.latent_std = kv_floats(gguf_, A + "latent_std");
    h.times = kv_floats(gguf_, A + "sampler_times");
    h.charset = kv_str(gguf_, A + "charset", "");
    h.weight_type = kv_str(gguf_, A + "weight_type", "?");
    h.rate_calibration = kv_strs(gguf_, A + "rate_calibration");
    if ((int) h.latent_mean.size() != h.latent || (int) h.latent_std.size() != h.latent || h.times.size() != 4) {
        err_ = "model metadata incomplete (latent statistics or sampler times)";
        return false;
    }

    fused_ = std::getenv("ARTICULINO_GENERIC_OPS") == nullptr;
    profile_ = std::getenv("ARTICULINO_PROFILE") != nullptr;
    backend_ = ggml_backend_cpu_init();
    if (!backend_) { err_ = "cannot initialise the CPU backend"; return false; }
    set_threads(n_threads);
    if (use_gpu) {
        // first GPU device ggml was built with (e.g. Vulkan); ops it cannot run fall back to the CPU (scheduler)
        ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gdev) gpu_ = ggml_backend_dev_init(gdev, nullptr);
        if (!gpu_) { err_ = "no GPU backend available (build with -DARTICULINO_VULKAN=ON)"; return false; }
        fused_ = false;              // the fused ops are CPU code
        profile_ = false;
        ggml_backend_t bs[2] = { gpu_, backend_ };
        sched_ = ggml_backend_sched_new(bs, nullptr, 2, 32768, false, false);
    }

    // Weights go into the default CPU buffer, except quantized matmul weights that an extra CPU buffer type
    // (the "repack" buffer: interleaved 4-bit layouts for AVX2/NEON GEMM) can take. Set ARTICULINO_NO_REPACK=1 to
    // keep everything in the default buffer. With a GPU, all weights go into GPU memory.
    ggml_backend_dev_t dev = ggml_backend_get_device(main_backend());
    std::vector<ggml_backend_buffer_type_t> extra;
    if (!std::getenv("ARTICULINO_NO_REPACK") && !gpu_) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra) for (ggml_backend_buffer_type_t * b = get_extra(dev); b && *b; b++) extra.push_back(*b);
    }
    std::vector<ggml_backend_buffer_type_t> bufts{ ggml_backend_dev_buffer_type(dev) };
    bufts.insert(bufts.end(), extra.begin(), extra.end());
    std::vector<ggml_backend_buffer_t> dummies;
    for (auto b : extra) dummies.push_back(ggml_backend_buft_alloc_buffer(b, 0));
    // does extra buffer type i support a matmul with this weight?
    auto extra_ok = [&](size_t i, ggml_tensor * t) {
        if (!ggml_is_quantized(t->type) || ggml_n_dims(t) != 2 || !dummies[i]) return false;
        ggml_init_params tp{ ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * tc = ggml_init(tp);
        ggml_tensor * w = ggml_dup_tensor(tc, t);
        ggml_tensor * x = ggml_new_tensor_2d(tc, GGML_TYPE_F32, t->ne[0], 16);
        ggml_tensor * op = ggml_mul_mat(tc, w, x);
        w->buffer = dummies[i];
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        w->buffer = nullptr;
        ggml_free(tc);
        return ok;
    };
    const int64_t n_tensors = gguf_get_n_tensors(gguf_);
    std::vector<int> where(n_tensors, 0);
    std::vector<int> count(bufts.size(), 0);
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(wctx_, gguf_get_tensor_name(gguf_, i));
        for (size_t e = 0; e < extra.size(); e++)
            if (extra_ok(e, t)) { where[i] = (int) e + 1; break; }
        count[where[i]]++;
    }
    for (auto d : dummies) if (d) ggml_backend_buffer_free(d);
    std::vector<ggml_context *> ctxs(bufts.size(), nullptr);
    for (size_t b = 0; b < bufts.size(); b++) {
        if (!count[b]) continue;
        ggml_init_params cp{ ggml_tensor_overhead() * (count[b] + 1), nullptr, true };
        ctxs[b] = ggml_init(cp);
        wctxs_.push_back(ctxs[b]);
    }
    // F16 weights are widened to F32 on the CPU: x86 has no F16 arithmetic, so ggml would convert the activations to
    // F16 for every matmul - slower and less accurate than F32 (ARTICULINO_KEEP_F16=1 keeps them as stored)
    const bool widen = !std::getenv("ARTICULINO_KEEP_F16") && !gpu_;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor * src = ggml_get_tensor(wctx_, name);
        ggml_tensor * t = (widen && src->type == GGML_TYPE_F16)
            ? ggml_new_tensor(ctxs[where[i]], GGML_TYPE_F32, ggml_n_dims(src), src->ne)
            : ggml_dup_tensor(ctxs[where[i]], src);
        ggml_set_name(t, name);
        w_[name] = t;
    }
    for (size_t b = 0; b < bufts.size(); b++) {
        if (!ctxs[b]) continue;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctxs[b], bufts[b]);
        if (!buf) { err_ = "cannot allocate the weight buffer"; return false; }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        wbufs_.push_back(buf);
    }
    n_repacked_ = (int) (n_tensors - count[0]);

    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err_ = "cannot open " + path; return false; }
    const size_t data_off = gguf_get_data_offset(gguf_);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_); i++) {
        const char * name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor * t = w_[name];
        const ggml_tensor * stored = ggml_get_tensor(wctx_, name);
        const size_t sz = ggml_nbytes(stored);
        buf.resize(sz);
#ifdef _WIN32
        _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(gguf_, i)), SEEK_SET);
#else
        fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(gguf_, i)), SEEK_SET);
#endif
        if (std::fread(buf.data(), 1, sz, f) != sz) { std::fclose(f); err_ = "short read in " + path; return false; }
        if (t->type != stored->type) {
            std::vector<float> wide((size_t) ggml_nelements(stored));
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), wide.data(), (int64_t) wide.size());
            ggml_backend_tensor_set(t, wide.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, buf.data(), 0, sz);
        }
    }
    std::fclose(f);

    mel_fb_ = get(W("audio.mel_fb"));
    hann_ = get(W("audio.hann"));
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    return true;
}

const char * Engine::device_name() const { return ggml_backend_name(main_backend()); }

void Engine::set_threads(int n) {
    if (n <= 0) n = 4;
    n_threads_ = n;
    if (!backend_) return;
    if (std::getenv("ARTICULINO_NO_THREADPOOL")) { ggml_backend_cpu_set_n_threads(backend_, n); return; }
    // a persistent thread pool: the workers stay alive between graphs instead of being created for every compute
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(backend_, tp);
    if (tp_) ggml_threadpool_free(tp_);
    tp_ = tp;
    ggml_backend_cpu_set_n_threads(backend_, n);
}

ggml_tensor * Engine::W(const std::string & name) const {
    auto it = w_.find(name);
    if (it == w_.end()) throw std::runtime_error("missing tensor in model: " + name);
    return it->second;
}

void Engine::compute(Graph & g) {
    if (sched_) {
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) throw std::runtime_error("graph allocation failed");
        for (auto & in : g.inputs) {
            if (!in.first->buffer) continue;
            ggml_backend_tensor_set(in.first, in.second.data(), 0, in.second.size());
        }
        if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        if (std::getenv("ARTICULINO_VERBOSE"))
            std::fprintf(stderr, "graph: %d nodes, %d splits, %d copies\n", ggml_graph_n_nodes(g.gf),
                         ggml_backend_sched_get_n_splits(sched_), ggml_backend_sched_get_n_copies(sched_));
        return;
    }
    if (!ggml_gallocr_alloc_graph(galloc_, g.gf)) throw std::runtime_error("graph allocation failed");
    for (auto & in : g.inputs) {
        if (!in.first->buffer) continue;     // input not used by this graph
        ggml_backend_tensor_set(in.first, in.second.data(), 0, in.second.size());
    }
    if (profile_) {   // ARTICULINO_PROFILE=1: run node by node and add up the time per op (slower, for analysis)
        ggml_init_params pp{ ggml_graph_overhead_custom(1, false) + 1024, nullptr, true };
        ggml_context * pc = ggml_init(pp);
        ggml_cgraph * one = ggml_new_graph_custom(pc, 1, false);
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
            ggml_tensor * node = ggml_graph_node(g.gf, i);
            ggml_graph_clear(one);
            ggml_graph_add_node(one, node);
            const double t0 = now_s();
            ggml_backend_graph_compute(backend_, one);
            std::string key = ggml_op_desc(node);
            if (node->op == GGML_OP_MUL_MAT) key += std::string(" ") + ggml_type_name(node->src[0]->type) + " " +
                std::to_string(node->src[0]->ne[0]) + "x" + std::to_string(node->src[0]->ne[1]);
            prof_[key].first += now_s() - t0;
            prof_[key].second++;
        }
        ggml_free(pc);
        return;
    }
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
}

void Engine::print_profile() const {
    std::vector<std::pair<double, std::string>> v;
    double total = 0;
    for (auto & kv : prof_) { v.emplace_back(kv.second.first, kv.first); total += kv.second.first; }
    std::sort(v.rbegin(), v.rend());
    std::fprintf(stderr, "profile (%.3f s in graph nodes):\n", total);
    for (auto & x : v) std::fprintf(stderr, "  %7.3f s %5.1f %%  %6d x  %s\n", x.first, 100 * x.first / total, prof_.at(x.second).second, x.second.c_str());
}

std::vector<float> Engine::get(ggml_tensor * t) const {
    std::vector<float> out((size_t) ggml_nelements(t));
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else {
        throw std::runtime_error(std::string("cannot read tensor ") + t->name);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ rng

static uint64_t splitmix64(uint64_t & x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

Rng::Rng(uint64_t seed) {
    uint64_t x = seed;
    for (auto & v : s) v = splitmix64(x);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t Rng::next() {   // xoshiro256**
    const uint64_t r = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
}

double Rng::uniform() {
    return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

void Rng::normal(float * out, size_t n) {   // Box-Muller, both outputs used
    size_t i = 0;
    while (i < n) {
        const double u1 = uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i++] = (float) (r * std::cos(2.0 * M_PI * u2));
        if (i < n) out[i++] = (float) (r * std::sin(2.0 * M_PI * u2));
    }
}

}  // namespace articulino
