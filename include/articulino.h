/*
 * articulino.cpp - C API
 *
 * German + English text to speech with zero-shot voice cloning (Articulino-DE-EN-Realtime-Stream-TTS),
 * running on ggml. Output is 24 kHz mono float audio in [-1, 1].
 *
 * Typical use:
 *     articulino_context * ctx = articulino_load("articulino-q8_0.gguf", NULL);
 *     articulino_set_voice_file(ctx, "voice.wav");               // 3-8 s, 24 kHz mono (others are resampled)
 *     articulino_options opt = articulino_default_options();
 *     articulino_say(ctx, "Guten Morgen! Wie geht es dir?", &opt, on_audio, user);   // one callback per sentence
 *     articulino_free(ctx);
 *
 * Text must already be spelled out (numbers, dates, abbreviations): the engine only lower-cases, strips accents and
 * keeps the 52 symbols the model knows. See README.md for the German text frontend that does the spelling.
 * A context is not thread safe; use one context per thread.
 */
#ifndef ARTICULINO_H
#define ARTICULINO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  ifdef ARTICULINO_BUILD_SHARED
#    define ARTICULINO_API __declspec(dllexport)
#  else
#    define ARTICULINO_API
#  endif
#else
#  define ARTICULINO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct articulino_context articulino_context;
typedef struct articulino_stream articulino_stream;

enum { ARTICULINO_LANG_DE = 0, ARTICULINO_LANG_EN = 1 };

typedef struct {
    int n_threads;          /* CPU threads for the compute graphs; <= 0 selects 4 */
    int use_gpu;            /* 1: run the graphs on the first GPU of a GPU-enabled build (Vulkan), CPU fallback */
} articulino_params;

typedef struct {
    int language;           /* ARTICULINO_LANG_DE or ARTICULINO_LANG_EN; "[en]..[/en]" in the text switches per word */
    float speed;            /* speaking rate, 0.7 .. 1.6; 1.0 = normal */
    uint64_t seed;          /* every sentence starts its noise from this seed: same seed, same audio */
    int equalize_rate;      /* 1: bring every sentence to the voice's mean letter rate (reference service); 0: raw */
    int postprocess;        /* 1: +2 dB low shelf at 180 Hz and peak limit 0.95 (reference service); 0: raw decoder */
    int retry_on_artifact;  /* 1: a sentence with latent outliers or a whistle is made once more with seed + 1 */
} articulino_options;

/* called once per sentence with its audio; return non-zero to stop */
typedef int (*articulino_audio_callback)(const float * samples, size_t n_samples, const char * sentence, void * user);

ARTICULINO_API articulino_params  articulino_default_params(void);
ARTICULINO_API articulino_options articulino_default_options(void);

/* load a model; NULL on failure (see articulino_last_error) */
ARTICULINO_API articulino_context * articulino_load(const char * model_path, const articulino_params * params);
ARTICULINO_API void articulino_free(articulino_context * ctx);
ARTICULINO_API int  articulino_sample_rate(const articulino_context * ctx);
ARTICULINO_API void articulino_set_threads(articulino_context * ctx, int n_threads);
/* last error message of the calling thread */
ARTICULINO_API const char * articulino_last_error(void);

/* set the voice from a WAV file or from samples (mono). The first 8 s are used; at least 3 s are recommended.
 * Computes the voice cache once; every later sentence reuses it. Returns 0 on success. */
ARTICULINO_API int articulino_set_voice_file(articulino_context * ctx, const char * wav_path);
ARTICULINO_API int articulino_set_voice(articulino_context * ctx, const float * samples, size_t n_samples, int sample_rate);

/* speak one sentence as given (no splitting). On success *samples is allocated (free with articulino_free_audio). */
ARTICULINO_API int articulino_say_sentence(articulino_context * ctx, const char * sentence, const articulino_options * opt,
                                           float ** samples, size_t * n_samples);
ARTICULINO_API void articulino_free_audio(float * samples);

/* speak a whole text: split into sentences like the reference service, callback after every sentence */
ARTICULINO_API int articulino_say(articulino_context * ctx, const char * text, const articulino_options * opt,
                                  articulino_audio_callback cb, void * user);

/* streaming input, e.g. tokens from a language model: every finished sentence is spoken at once */
ARTICULINO_API articulino_stream * articulino_stream_begin(articulino_context * ctx, const articulino_options * opt,
                                                           articulino_audio_callback cb, void * user);
ARTICULINO_API int articulino_stream_push(articulino_stream * stream, const char * text_piece);
/* speak what is left and free the stream */
ARTICULINO_API int articulino_stream_end(articulino_stream * stream);

/* the letters the model will see for a text (after normalization); free with articulino_free_string */
ARTICULINO_API char * articulino_normalize(const char * text);
ARTICULINO_API void articulino_free_string(char * s);

#ifdef __cplusplus
}
#endif

#endif /* ARTICULINO_H */
