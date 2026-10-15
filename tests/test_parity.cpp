// Parity test against the reference vectors of the original PyTorch code (tests/vectors/*.npz, exported with
// tests/export_vectors.py). Every building block gets the reference input of its stage, so errors do not add up;
// at the end the whole chain runs from the voice WAV and the sentence with the reference noise.
//
//   test-parity model.gguf de_sentence.vectors.gguf [--threads N] [--tol 1e-3]
//
// Prints max|d|/max|ref| and SNR per stage; exit code 1 if an F32 model misses the tolerance of docs/SPEC.md.
#include "engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace articulino;

struct Vectors {
    gguf_context * g = nullptr;
    ggml_context * ctx = nullptr;
    bool load(const std::string & p) {
        gguf_init_params ip{ false, &ctx };
        g = gguf_init_from_file(p.c_str(), ip);
        return g != nullptr;
    }
    ggml_tensor * t(const std::string & name) const {
        ggml_tensor * x = ggml_get_tensor(ctx, name.c_str());
        if (!x) { std::fprintf(stderr, "missing vector %s\n", name.c_str()); std::exit(2); }
        return x;
    }
    std::vector<float> f(const std::string & name) const {
        ggml_tensor * x = t(name);
        const float * d = (const float *) x->data;
        return std::vector<float>(d, d + ggml_nelements(x));
    }
    std::vector<int32_t> i(const std::string & name) const {
        ggml_tensor * x = t(name);
        const int32_t * d = (const int32_t *) x->data;
        return std::vector<int32_t>(d, d + ggml_nelements(x));
    }
    std::string s(const char * key) const {
        const int64_t id = gguf_find_key(g, key);
        return id < 0 ? "" : gguf_get_val_str(g, id);
    }
};

// numpy (a, b) row-major -> (b, a)
static std::vector<float> transpose(const std::vector<float> & x, int a, int b) {
    std::vector<float> y(x.size());
    for (int i = 0; i < a; i++)
        for (int j = 0; j < b; j++) y[(size_t) j * a + i] = x[(size_t) i * b + j];
    return y;
}

static int n_fail = 0;
static double g_tol = 1e-3;
static bool g_strict = true;

// mode 0: block, max|d|/max|ref| < tol; 1: audio, SNR > 40 dB; 2: information only
static void compare(const char * name, const std::vector<float> & ref, const std::vector<float> & got, int mode = 0) {
    if (ref.size() != got.size()) {
        std::printf("%-22s SIZE MISMATCH ref %zu got %zu\n", name, ref.size(), got.size());
        n_fail++;
        return;
    }
    double maxref = 0, maxd = 0, sref = 0, sd = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) got[i] - ref[i];
        maxref = std::max(maxref, std::fabs((double) ref[i]));
        maxd = std::max(maxd, std::fabs(d));
        sref += (double) ref[i] * ref[i];
        sd += d * d;
    }
    const double rel = maxref > 0 ? maxd / maxref : maxd;
    const double snr = sd > 0 ? 10.0 * std::log10(sref / sd) : 999.0;
    const bool ok = mode == 1 ? snr > 40.0 : (mode == 2 || rel < g_tol);
    if (!ok && g_strict) n_fail++;
    std::printf("%-22s n=%-8zu max|d|/max|ref| %.3e  SNR %6.1f dB  %s\n", name, ref.size(), rel, snr,
                mode == 2 ? "(info)" : ok ? "ok" : (g_strict ? "FAIL" : "-"));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test-parity model.gguf vectors.gguf [--threads N] [--tol X]\n");
        return 2;
    }
    int threads = 4;
    bool gpu = false;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--gpu")) gpu = true;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) g_tol = std::atof(argv[++i]);
    }
    Engine eng;
    if (!eng.load(argv[1], threads, gpu)) { std::fprintf(stderr, "error: %s\n", eng.error().c_str()); return 1; }
    Vectors V;
    if (!V.load(argv[2])) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    const HParams & hp = eng.hp();
    g_strict = hp.weight_type == "f32";
    const int lat = hp.latent, D = hp.text_dim;
    std::printf("model %s (%s) on %s, vectors %s: \"%s\" (%s)\n", argv[1], hp.weight_type.c_str(), eng.device_name(), argv[2],
                V.s("vectors.sentence").c_str(), V.s("vectors.language").c_str());

    // 1. log-mel
    const std::vector<float> wav = V.f("voice_wav");
    int frames = 0;
    std::vector<float> mel = eng.logmel(wav.data(), (int) wav.size(), frames);
    const std::vector<float> mel_ref = V.f("voice_logmel");                 // (100, F)
    compare("voice_logmel", transpose(mel_ref, hp.n_mels, frames), mel);
    // 2. latent encoder on the reference mel
    const int use = (int) wav.size() / hp.mel_hop;
    std::vector<float> mel_in = transpose(mel_ref, hp.n_mels, frames);
    mel_in.resize((size_t) use * hp.n_mels);
    int n_lat = 0;
    std::vector<float> mu = eng.encode_latents(mel_in, use, n_lat);
    compare("voice_mu", V.f("voice_mu"), transpose(mu, n_lat, lat));
    // 3. voice encoder on the reference latents
    Voice v;
    v.n_ref = n_lat;
    v.ref = V.f("voice_ref");
    std::vector<float> tokens, vec;
    eng.voice_encode(v.ref, v.n_ref, tokens, vec);
    compare("voice_tokens", V.f("voice_tokens"), tokens);
    compare("voice_vector", V.f("voice_vector"), vec);
    // 4. prompt cache from the reference voice
    v.tokens = V.f("voice_tokens");
    v.vec = V.f("voice_vector");
    if (!eng.build_prompt_cache(v)) { std::fprintf(stderr, "%s\n", eng.error().c_str()); return 1; }
    const int P = hp.prompt_frames, H = hp.dit_heads, dh = hp.dit_dim / hp.dit_heads;
    {
        std::vector<float> kr, vr, ka, va;
        for (int b = 0; b < hp.dit_layers; b++) {
            std::vector<float> k, vv;
            eng.read_prompt_cache(v, b, k, vv);                                  // [P][H][dh]
            std::vector<float> kt(k.size()), vt(vv.size());                      // -> [H][P][dh]
            for (int p = 0; p < P; p++)
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < dh; d++) {
                        kt[((size_t) h * P + p) * dh + d] = k[((size_t) p * H + h) * dh + d];
                        vt[((size_t) h * P + p) * dh + d] = vv[((size_t) p * H + h) * dh + d];
                    }
            const std::vector<float> rk = V.f("cache_k" + std::to_string(b)), rv = V.f("cache_v" + std::to_string(b));
            kr.insert(kr.end(), rk.begin(), rk.end()); vr.insert(vr.end(), rv.begin(), rv.end());
            ka.insert(ka.end(), kt.begin(), kt.end()); va.insert(va.end(), vt.begin(), vt.end());
            if (b == 0 || b == hp.dit_layers - 1) {
                compare(("cache_k" + std::to_string(b)).c_str(), rk, kt);
                compare(("cache_v" + std::to_string(b)).c_str(), rv, vt);
            }
        }
        compare("cache_k (all)", kr, ka);
        compare("cache_v (all)", vr, va);
    }
    // 5. text encoder + duration predictor
    const std::vector<int32_t> ids = V.i("text_ids");
    const std::vector<float> mask = V.f("text_mask");
    const int lang = V.s("vectors.language") == "en" ? 1 : 0;
    std::vector<int32_t> langs(ids.size(), lang);
    std::vector<float> h, dl;
    eng.text_and_durations(ids, langs, mask, v, h, dl);
    compare("text_h", V.f("text_h"), h);
    compare("duration_log", V.f("duration_log"), dl);
    // 6. durations (raw, as the vectors: no rate equalization) and upsampling
    SynthOptions o;
    o.equalize_rate = false;
    o.postprocess = false;
    o.language = lang;
    std::vector<float> d = eng.durations(V.f("duration_log"), mask, ids, v, o);
    compare("durations", V.f("durations"), d);
    const std::vector<float> fc_ref = V.f("frame_cond");
    const int L = (int) (fc_ref.size() / D);
    std::vector<float> hf = eng.upsample(V.f("text_h"), V.f("durations"), L);
    compare("frame_cond", fc_ref, hf);
    // 7. sampler: every step from its reference input
    const int Ln = L - P;
    std::vector<float> hf_new(fc_ref.begin() + (size_t) P * D, fc_ref.end());
    std::vector<float> noise[4];
    for (int k = 0; k < 4; k++) noise[k] = V.f("noise_" + std::to_string(k));
    NoiseFn nf = [&](int k, int n, int l, float * dst) { std::memcpy(dst, noise[k].data(), (size_t) n * l * sizeof(float)); };
    std::vector<float> forced[4];
    for (int k = 0; k < 4; k++) forced[k] = V.f("step" + std::to_string(k) + "_in");
    SynthTrace tr;
    eng.sample(v, hf_new, Ln, nf, &tr, forced);
    for (int k = 0; k < 4; k++) {
        const std::string s = "step" + std::to_string(k);
        compare((s + "_velocity").c_str(), V.f(s + "_velocity"), tr.step_v[k]);
        compare((s + "_x1").c_str(), V.f(s + "_x1"), tr.step_x1[k]);
    }
    // chained (as in use): step inputs follow from the previous step
    SynthTrace tc;
    std::vector<float> x1 = eng.sample(v, hf_new, Ln, nf, &tc);
    for (int k = 1; k < 4; k++) {
        const std::string s = "step" + std::to_string(k) + "_in (chained)";
        compare(s.c_str(), V.f("step" + std::to_string(k) + "_in"), tc.step_in[k]);
    }
    compare("latents (chained)", V.f("step3_x1"), x1);
    // 8. decoder on the reference input
    const std::vector<float> din = V.f("decoder_in");                       // (96, T)
    const int T = (int) (din.size() / lat);
    std::vector<float> wav_out = eng.decode(transpose(din, lat, T), T);
    compare("decoder_out", V.f("decoder_out"), wav_out);
    compare("decoder_out (SNR)", V.f("decoder_out"), wav_out, 1);
    auto mel_dist = [&](const std::vector<float> & a, const std::vector<float> & b) {
        int fa = 0, fb = 0;
        const std::vector<float> ma = eng.logmel(a.data(), (int) a.size(), fa), mb = eng.logmel(b.data(), (int) b.size(), fb);
        double sum = 0;
        for (size_t i = 0; i < ma.size() && i < mb.size(); i++) sum += std::fabs(ma[i] - mb[i]);
        return 20.0 / std::log(10.0) * sum / std::max<size_t>(1, ma.size());
    };
    std::printf("%-22s mean |d log-mel| %.3f dB\n", "decoder_out (spectrum)", mel_dist(V.f("decoder_out"), wav_out));
    // 9. whole chain from the voice WAV and the sentence, reference noise, no post-processing
    Voice v2;
    if (!eng.make_voice(wav.data(), (int) wav.size(), v2)) { std::fprintf(stderr, "%s\n", eng.error().c_str()); return 1; }
    compare("voice_ref (from wav)", V.f("voice_ref"), v2.ref);
    SynthTrace tf;
    std::vector<float> audio = eng.synthesize(v2, V.s("vectors.sentence"), [&] { SynthOptions q = o; q.noise = nf; return q; }(), &tf);
    if (tf.ids != ids) { std::printf("text ids differ from the reference\n"); n_fail++; }
    if (tf.frames_total != L) { std::printf("frame count %d differs from the reference %d\n", tf.frames_total, L); n_fail++; }
    else {
        compare("chain: latents", V.f("latents_norm"), tf.latents_norm);
        compare("chain: audio (SNR)", V.f("audio"), audio, 1);
        compare("chain: audio (max)", V.f("audio"), audio, 2);
        // spectral distance: mean |log-mel difference| in dB (ignores phase, closer to what one hears)
        std::printf("%-22s mean |d log-mel| %.3f dB\n", "chain: spectrum", mel_dist(V.f("audio"), audio));
    }
    std::printf("%s\n", n_fail ? "FAILED" : (g_strict ? "all stages within tolerance" : "done (quantized: SNR only)"));
    return n_fail ? 1 : 0;
}
