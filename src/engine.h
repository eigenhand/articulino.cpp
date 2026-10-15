// Internal C++ API of articulino.cpp. The public interface is the C API in include/articulino.h.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace articulino {

struct HParams {
    int sample_rate = 24000;
    int vocab = 53;
    int latent = 96;
    int samples_per_latent = 960;
    int units_per_latent = 4;          // durations are in 10 ms units, 4 per latent frame
    float duration_max = 400.0f;
    int prompt_frames = 125;           // voice prompt in front of every sentence (5 s)
    int decoder_context = 12;          // prompt frames decoded in front of a sentence and cut off
    int voice_max_samples = 8 * 24000;
    int text_dim = 512, text_layers = 6, text_heads = 8;
    int voice_dim = 512, voice_layers = 4, voice_heads = 8, voice_tokens = 32, voice_out = 768;
    int dur_width = 384;
    int dit_dim = 384, dit_layers = 8, dit_heads = 6;
    float rope_base = 10000.0f;
    int mel_nfft = 1024, mel_win = 960, mel_hop = 240, n_mels = 100;
    int istft_nfft = 960, istft_hop = 240;
    int enc_dim = 512, enc_blocks = 12, dec_dim = 512, dec_blocks = 12;
    float shelf_db = 2.0f, shelf_hz = 180.0f, shelf_slope = 0.7f, peak = 0.95f;
    std::vector<float> latent_mean, latent_std, times;
    std::string charset;
    std::string weight_type;
    std::vector<std::string> rate_calibration;
};

// one forward graph; host input data is stashed and uploaded after allocation
struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> inputs;

    explicit Graph(size_t n_nodes = 4096);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    ggml_tensor * in_f32(const float * data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1);
    ggml_tensor * in_i32(const int32_t * data, int64_t ne0);
    void out(ggml_tensor * t);       // keep this tensor's data after compute
    void expand(ggml_tensor * t);    // add a side effect node (e.g. a copy into a persistent tensor)
};

// the per-voice state: prompt latents and everything the DiT needs from the voice, computed once
struct Voice {
    int n_ref = 0;                     // reference latent frames
    std::vector<float> ref;            // [n_ref][latent] normalized latents
    std::vector<float> tokens;         // [voice_tokens][voice_out]
    std::vector<float> vec;            // [voice_out]
    float base_rate = -1.0f;           // letters per second at speed 1 (for rate equalization), lazily computed
    // persistent backend tensors
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int n_prompt = 0;                  // prompt frames in front of every sentence (5 s, fewer for short voices)
    std::vector<ggml_tensor *> kp, vt;   // prompt keys [dh, prompt, heads] / values [prompt, dh, heads] per DiT block
    std::vector<ggml_tensor *> xkp, xvt; // cross-attention keys [dh, tokens, heads] / values [tokens, dh, heads]
    ggml_tensor * ada = nullptr;       // AdaNorm (1 + scale, shift) per block and sampler step [dit_dim, 4 L + 2, 4]
    ggml_tensor * g_raw = nullptr;     // [voice_out] voice vector (duration predictor)

    Voice() = default;
    Voice(const Voice &) = delete;
    Voice & operator=(const Voice &) = delete;
    ~Voice();
};

// AdaNorm parameters of one DiT block
struct DitMod {
    ggml_tensor * attn_scale1p = nullptr, * attn_shift = nullptr, * ffn_scale1p = nullptr, * ffn_shift = nullptr;
};

// noise for the 4 sampler steps: called with (step, n_frames, latent) and must fill n_frames*latent floats
using NoiseFn = std::function<void(int step, int n_frames, int latent, float * out)>;

struct SynthOptions {
    float speed = 1.0f;
    uint64_t seed = 0;
    bool equalize_rate = true;         // bring every sentence to the voice's mean letter rate (service default)
    bool postprocess = true;           // low shelf + peak limit
    bool retry_on_artifact = true;     // remake a broken sentence once with seed + 1 (as the service; never with noise)
    int language = 0;                  // 0 = de, 1 = en (base language; [en]..[/en] markup switches per letter)
    NoiseFn noise;                     // null: own generator from seed
};

// intermediate results of one sentence, filled when requested (tests, debugging)
struct SynthTrace {
    std::vector<int32_t> ids;          // with the leading prompt placeholder
    std::vector<float> text_h;         // [n][text_dim]
    std::vector<float> dur_log;        // [n]
    std::vector<float> durations;      // [n]
    std::vector<float> frame_cond;     // [L][text_dim]
    int frames_total = 0, frames_new = 0;
    std::vector<float> step_in[4], step_v[4], step_x1[4];  // [new][latent]
    std::vector<float> latents_norm;   // [prompt + new][latent]
    std::vector<float> decoder_out;    // waveform incl. context
    std::vector<float> audio;          // before post-processing
    std::string artifact;              // why the first attempt was remade ('' = it was not)
};

struct Timing {
    double text = 0, dit = 0, decoder = 0, istft = 0, post = 0, total = 0;
    int retries = 0;
};

class Engine {
public:
    Engine() = default;
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    bool load(const std::string & gguf_path, int n_threads, bool use_gpu = false);
    const char * device_name() const;
    void set_threads(int n);
    int threads() const { return n_threads_; }
    int repacked_tensors() const { return n_repacked_; }
    const HParams & hp() const { return hp_; }
    const std::string & error() const { return err_; }

    // ---- voice: 24 kHz mono samples -> voice state (latent encoder, voice encoder, prompt cache)
    bool make_voice(const float * wav, int n, Voice & v);

    // ---- building blocks (exposed for the parity tests)
    std::vector<float> logmel(const float * wav, int n, int & frames) const;              // [frames][n_mels]
    std::vector<float> encode_latents(const std::vector<float> & mel, int frames, int & n_lat);  // mu [n_lat][latent]
    void voice_encode(const std::vector<float> & ref, int n_ref, std::vector<float> & tokens, std::vector<float> & vec);
    bool build_prompt_cache(Voice & v);
    void read_prompt_cache(const Voice & v, int block, std::vector<float> & k, std::vector<float> & vv) const;
    void text_and_durations(const std::vector<int32_t> & ids, const std::vector<int32_t> & lang,
                            const std::vector<float> & mask, const Voice & v,
                            std::vector<float> & h, std::vector<float> & dur_log);
    std::vector<float> durations(const std::vector<float> & dur_log, const std::vector<float> & mask,
                                 const std::vector<int32_t> & ids, const Voice & v, const SynthOptions & o);
    std::vector<float> upsample(const std::vector<float> & h, const std::vector<float> & dur, int n_frames) const;
    // 4-step sampler on the new frames; hf_new [L][text_dim] -> latents [L][latent] (normalized)
    // forced_in (tests): 4 arrays used as the input of each step instead of the chained re-noising
    std::vector<float> sample(const Voice & v, const std::vector<float> & hf_new, int L, const NoiseFn & noise,
                              SynthTrace * trace, const std::vector<float> * forced_in = nullptr);
    // decoder: z [T][latent] (de-normalized) -> waveform T*960
    std::vector<float> decode(const std::vector<float> & z, int T);

    // ---- one sentence of already split text; returns 24 kHz audio
    std::vector<float> synthesize(Voice & v, const std::string & sentence, const SynthOptions & o,
                                  SynthTrace * trace = nullptr, Timing * timing = nullptr);

    float voice_base_rate(Voice & v);
    std::string artifact(const std::vector<float> & z, const float * wav, size_t n) const;
    void print_profile() const;

private:
    HParams hp_;
    std::string err_;
    int n_threads_ = 4;
    gguf_context * gguf_ = nullptr;
    ggml_context * wctx_ = nullptr;
    ggml_backend_t backend_ = nullptr;   // CPU (always there)
    ggml_backend_t gpu_ = nullptr;       // optional GPU backend; graphs then run through the scheduler
    ggml_backend_sched_t sched_ = nullptr;
    ggml_backend_t main_backend() const { return gpu_ ? gpu_ : backend_; }
    struct ggml_threadpool * tp_ = nullptr;
    std::vector<ggml_backend_buffer_t> wbufs_;   // weight buffers (default CPU, and the repack buffer for 4-bit)
    std::vector<ggml_context *> wctxs_;
    int n_repacked_ = 0;
    ggml_gallocr_t galloc_ = nullptr;
    std::map<std::string, ggml_tensor *> w_;
    std::vector<float> mel_fb_;    // [n_freq][n_mels]
    std::vector<float> hann_;      // [win]

    bool profile_ = false;
    std::map<std::string, std::pair<double, int>> prof_;
    bool fused_ = true;            // fused CPU custom ops (depthwise conv, GELU); off: plain ggml ops
    ggml_tensor * W(const std::string & name) const;
    ggml_tensor * gelu(ggml_context * c, ggml_tensor * x) const;
    void compute(Graph & g);
    std::vector<float> get(ggml_tensor * t) const;

    // graph builders
    ggml_tensor * tblock(Graph & g, const std::string & p, ggml_tensor * x, ggml_tensor * pos,
                         ggml_tensor * mask, int heads);
    ggml_tensor * convnext(Graph & g, const std::string & p, ggml_tensor * x);
    ggml_tensor * dit_block(Graph & g, int i, ggml_tensor * x, const DitMod & m, ggml_tensor * pos,
                            ggml_tensor * kp_prev, ggml_tensor * vt_prev, ggml_tensor * xkp, ggml_tensor * xvt,
                            ggml_tensor ** kp_out, ggml_tensor ** vt_out);
    ggml_tensor * time_cond(Graph & g, float t, ggml_tensor * g_dit);
    ggml_tensor * rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos) const;
    std::vector<float> istft(const std::vector<float> & head, int n_frames, int length) const;
};

// ---- text (text.cpp)
std::string normalize_text(const std::string & utf8);
// ids + per-letter language from text with optional [en]..[/en] / [de]..[/de] markup
void text_to_ids(const std::string & utf8, int base_lang, std::vector<int32_t> & ids, std::vector<int32_t> & lang);
// split a growing text buffer into sentences like the reference service; `all` flushes the rest
std::vector<std::string> split_sentences(const std::string & buffer, bool all, std::string & rest);
bool is_alnum_symbol(int32_t id);
// language markup still open after this text: -1 none, 0 [de], 1 [en] (state = the one open before it)
int markup_language_after(const std::string & s, int state);

// ---- audio (audio.cpp)
bool read_wav(const std::string & path, std::vector<float> & mono, int & sample_rate, std::string & err);
bool write_wav(const std::string & path, const float * x, size_t n, int sample_rate);
std::vector<float> resample(const std::vector<float> & x, int sr_in, int sr_out);
void low_shelf(std::vector<float> & x, float f0, float db, int sr, float slope);
// a narrow-band tone above 1.8 kHz holding >= 60 ms (the service's whistle detector)
bool narrow_band_tone(const float * x, size_t n, int sr, float & ms, float & hz, int n_threads = 1);

// ---- noise
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    double uniform();          // (0, 1)
    void normal(float * out, size_t n);
};

double now_s();

}  // namespace articulino
