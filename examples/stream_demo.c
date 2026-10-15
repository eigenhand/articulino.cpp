/* Minimal C example: stream text piece by piece (as from a language model) and write each sentence's audio to a
 * raw WAV file as soon as it is ready.
 *
 *   stream-demo model.gguf voice.wav out.wav
 */
#include "articulino.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { FILE * f; size_t n; } sink;

static void put32(FILE * f, unsigned v) { unsigned char b[4] = { v, v >> 8, v >> 16, v >> 24 }; fwrite(b, 1, 4, f); }
static void put16(FILE * f, unsigned v) { unsigned char b[2] = { v, v >> 8 }; fwrite(b, 1, 2, f); }

static void wav_header(FILE * f, size_t n) {
    fwrite("RIFF", 1, 4, f); put32(f, (unsigned) (36 + n * 2)); fwrite("WAVEfmt ", 1, 8, f);
    put32(f, 16); put16(f, 1); put16(f, 1); put32(f, 24000); put32(f, 48000); put16(f, 2); put16(f, 16);
    fwrite("data", 1, 4, f); put32(f, (unsigned) (n * 2));
}

static int on_audio(const float * x, size_t n, const char * sentence, void * user) {
    sink * s = (sink *) user;
    for (size_t i = 0; i < n; i++) {
        float v = x[i] < -1.0f ? -1.0f : (x[i] > 1.0f ? 1.0f : x[i]);
        short p = (short) (v * 32767.0f);
        fwrite(&p, 2, 1, s->f);
    }
    s->n += n;
    fprintf(stderr, "sentence ready: %.2f s  \"%s\"\n", n / 24000.0, sentence);
    return 0;   /* non-zero would stop */
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf voice.wav out.wav\n", argv[0]); return 2; }
    articulino_params p = articulino_default_params();
    p.n_threads = 4;
    articulino_context * ctx = articulino_load(argv[1], &p);
    if (!ctx) { fprintf(stderr, "load failed: %s\n", articulino_last_error()); return 1; }
    if (articulino_set_voice_file(ctx, argv[2]) != 0) { fprintf(stderr, "voice: %s\n", articulino_last_error()); return 1; }

    sink s = { fopen(argv[3], "wb"), 0 };
    wav_header(s.f, 0);
    articulino_options o = articulino_default_options();
    o.seed = 7;
    articulino_stream * st = articulino_stream_begin(ctx, &o, on_audio, &s);
    const char * pieces[] = { "Guten Mor", "gen! Heute ", "regnet es ab achtzehn Uhr. ", "Nimm einen Schirm mit, ",
                              "und denk an den [en]Meeting[/en]-Termin um neun." };
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) articulino_stream_push(st, pieces[i]);
    articulino_stream_end(st);

    fseek(s.f, 0, SEEK_SET);
    wav_header(s.f, s.n);
    fclose(s.f);
    char * norm = articulino_normalize("Café „Olé“ – Straße ÄÖÜ!");
    fprintf(stderr, "normalized: %s\n", norm);
    articulino_free_string(norm);
    articulino_free(ctx);
    return 0;
}
