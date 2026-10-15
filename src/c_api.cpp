// C API (include/articulino.h) on top of the C++ engine.
#include "articulino.h"
#include "engine.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

using namespace articulino;

struct articulino_context {
    Engine engine;
    std::unique_ptr<Voice> voice;
};

struct articulino_stream {
    articulino_context * ctx;
    articulino_options opt;
    articulino_audio_callback cb;
    void * user;
    std::string buffer;
    int lang_state;       // language still open from a previous sentence ([en] without [/en]), -1 = none
    bool stopped;
};

static thread_local std::string g_error;

static int fail(const std::string & msg) {
    g_error = msg;
    return -1;
}

static SynthOptions to_options(const articulino_options * o) {
    const articulino_options d = articulino_default_options();
    if (!o) o = &d;
    SynthOptions s;
    s.language = o->language == ARTICULINO_LANG_EN ? 1 : 0;
    s.speed = o->speed > 0.0f ? o->speed : 1.0f;
    s.speed = s.speed < 0.7f ? 0.7f : (s.speed > 1.6f ? 1.6f : s.speed);
    s.seed = o->seed;
    s.equalize_rate = o->equalize_rate != 0;
    s.postprocess = o->postprocess != 0;
    s.retry_on_artifact = o->retry_on_artifact != 0;
    return s;
}

static int speak_sentences(articulino_context * ctx, const std::vector<std::string> & sentences, const SynthOptions & o,
                           articulino_audio_callback cb, void * user, int & lang_state) {
    for (const std::string & s0 : sentences) {
        std::string s = s0;
        if (lang_state == 1) s = "[en]" + s;
        else if (lang_state == 0) s = "[de]" + s;
        lang_state = markup_language_after(s0, lang_state);
        std::vector<float> wav = ctx->engine.synthesize(*ctx->voice, s, o);
        if (wav.empty()) continue;
        if (cb && cb(wav.data(), wav.size(), s0.c_str(), user) != 0) return 1;
    }
    return 0;
}

extern "C" {

articulino_params articulino_default_params(void) {
    articulino_params p;
    p.n_threads = 4;
    p.use_gpu = 0;
    return p;
}

articulino_options articulino_default_options(void) {
    articulino_options o;
    o.language = ARTICULINO_LANG_DE;
    o.speed = 1.0f;
    o.seed = 0;
    o.equalize_rate = 1;
    o.postprocess = 1;
    o.retry_on_artifact = 1;
    return o;
}

articulino_context * articulino_load(const char * model_path, const articulino_params * params) {
    try {
        const articulino_params d = articulino_default_params();
        if (!params) params = &d;
        auto * ctx = new articulino_context();
        if (!ctx->engine.load(model_path ? model_path : "", params->n_threads, params->use_gpu != 0)) {
            g_error = ctx->engine.error();
            delete ctx;
            return nullptr;
        }
        return ctx;
    } catch (const std::exception & e) {
        g_error = e.what();
        return nullptr;
    }
}

void articulino_free(articulino_context * ctx) { delete ctx; }

int articulino_sample_rate(const articulino_context * ctx) { return ctx ? ctx->engine.hp().sample_rate : 24000; }

void articulino_set_threads(articulino_context * ctx, int n_threads) {
    if (ctx) ctx->engine.set_threads(n_threads);
}

const char * articulino_last_error(void) { return g_error.c_str(); }

int articulino_set_voice(articulino_context * ctx, const float * samples, size_t n_samples, int sample_rate) {
    if (!ctx || !samples) return fail("invalid arguments");
    try {
        std::vector<float> x(samples, samples + n_samples);
        const int sr = ctx->engine.hp().sample_rate;
        if (sample_rate != sr) x = resample(x, sample_rate, sr);
        auto v = std::make_unique<Voice>();
        if (!ctx->engine.make_voice(x.data(), (int) x.size(), *v)) return fail(ctx->engine.error());
        ctx->engine.voice_base_rate(*v);       // rate calibration now, not in the first sentence
        ctx->voice = std::move(v);
        return 0;
    } catch (const std::exception & e) {
        return fail(e.what());
    }
}

int articulino_set_voice_file(articulino_context * ctx, const char * wav_path) {
    if (!ctx || !wav_path) return fail("invalid arguments");
    std::vector<float> x;
    int sr = 0;
    std::string err;
    if (!read_wav(wav_path, x, sr, err)) return fail(err);
    return articulino_set_voice(ctx, x.data(), x.size(), sr);
}

int articulino_say_sentence(articulino_context * ctx, const char * sentence, const articulino_options * opt,
                            float ** samples, size_t * n_samples) {
    if (!ctx || !sentence || !samples || !n_samples) return fail("invalid arguments");
    if (!ctx->voice) return fail("no voice set");
    try {
        std::vector<float> wav = ctx->engine.synthesize(*ctx->voice, sentence, to_options(opt));
        *n_samples = wav.size();
        *samples = (float *) std::malloc(std::max<size_t>(1, wav.size()) * sizeof(float));
        if (!*samples) return fail("out of memory");
        if (!wav.empty()) std::memcpy(*samples, wav.data(), wav.size() * sizeof(float));
        return 0;
    } catch (const std::exception & e) {
        return fail(e.what());
    }
}

void articulino_free_audio(float * samples) { std::free(samples); }

int articulino_say(articulino_context * ctx, const char * text, const articulino_options * opt,
                   articulino_audio_callback cb, void * user) {
    if (!ctx || !text) return fail("invalid arguments");
    if (!ctx->voice) return fail("no voice set");
    try {
        std::string rest;
        const std::vector<std::string> sentences = split_sentences(text, true, rest);
        int state = -1;
        const int r = speak_sentences(ctx, sentences, to_options(opt), cb, user, state);
        return r < 0 ? r : 0;
    } catch (const std::exception & e) {
        return fail(e.what());
    }
}

articulino_stream * articulino_stream_begin(articulino_context * ctx, const articulino_options * opt,
                                            articulino_audio_callback cb, void * user) {
    if (!ctx) { fail("invalid arguments"); return nullptr; }
    if (!ctx->voice) { fail("no voice set"); return nullptr; }
    auto * s = new articulino_stream();
    s->ctx = ctx;
    s->opt = opt ? *opt : articulino_default_options();
    s->cb = cb;
    s->user = user;
    s->lang_state = -1;
    s->stopped = false;
    return s;
}

static int stream_run(articulino_stream * s, bool all) {
    if (s->stopped) return 1;
    try {
        std::string rest;
        const std::vector<std::string> sentences = split_sentences(s->buffer, all, rest);
        s->buffer = rest;
        const int r = speak_sentences(s->ctx, sentences, to_options(&s->opt), s->cb, s->user, s->lang_state);
        if (r == 1) s->stopped = true;
        return r;
    } catch (const std::exception & e) {
        return fail(e.what());
    }
}

int articulino_stream_push(articulino_stream * s, const char * text_piece) {
    if (!s) return fail("invalid arguments");
    if (text_piece) s->buffer += text_piece;
    return stream_run(s, false);
}

int articulino_stream_end(articulino_stream * s) {
    if (!s) return fail("invalid arguments");
    const int r = stream_run(s, true);
    delete s;
    return r < 0 ? r : 0;
}

char * articulino_normalize(const char * text) {
    const std::string n = normalize_text(text ? text : "");
    char * out = (char *) std::malloc(n.size() + 1);
    if (out) std::memcpy(out, n.c_str(), n.size() + 1);
    return out;
}

void articulino_free_string(char * s) { std::free(s); }

}  // extern "C"
