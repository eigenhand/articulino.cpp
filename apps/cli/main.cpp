// articulino-cli: text to speech from the command line.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace articulino;

static void usage() {
    std::fprintf(stderr,
        "usage: articulino-cli -m model.gguf --voice voice.wav (-t \"text\" | -f text.txt) -o out.wav [options]\n"
        "\n"
        "options:\n"
        "  -m, --model PATH     GGUF model file\n"
        "  --voice PATH         voice reference, WAV (3-8 s of clean speech; first 8 s are used)\n"
        "  -t, --text TEXT      text to speak (already spelled out: numbers etc. as words)\n"
        "  -f, --file PATH      read the text from a UTF-8 file\n"
        "  -o, --out PATH       output WAV (24 kHz mono, 16 bit); default out.wav; '-' = raw 16-bit PCM to stdout,\n"
        "                       written sentence by sentence (pipe into a player)\n"
        "  --stdin              read the text from stdin as it arrives and speak each sentence once it is complete\n"
        "  --threads N          CPU threads (default 4)\n"
        "  --gpu                run the graphs on the first GPU ggml was built with (Vulkan build), CPU fallback\n"
        "  --seed S             noise seed (default 0); every sentence starts from it\n"
        "  --speed X            speaking rate 0.7..1.6 (default 1.0)\n"
        "  --lang de|en         base language (default de); [en]..[/en] in the text switches per word\n"
        "  --pause MS           silence between sentences in ms (default 0)\n"
        "  --no-rate-eq         keep the raw durations (no per-voice letter rate equalization)\n"
        "  --no-post            no low shelf / peak limit after the decoder\n"
        "  --no-retry           keep a sentence even if it has latent outliers or a whistle (default: remake once)\n"
        "  --sentence           speak the text as one sentence (no splitting)\n"
        "  --noise PATH         test mode: noise from a .npy file, float32 (4, frames, 96); implies --sentence\n"
        "  --normalize          only print the normalized letters and the sentence split, then exit\n"
        "  --bench PATH         real-time factor over the sentences in PATH (one per line), like rtf.py\n"
        "  -q, --quiet          less output\n");
}

// minimal .npy reader for little-endian float32 C-order arrays
static bool read_npy(const std::string & path, std::vector<float> & data, std::vector<size_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t ver[2];
    f.read((char *) ver, 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char *) &h, 2); hlen = h; }
    else { f.read((char *) &hlen, 4); }
    std::string header(hlen, ' ');
    f.read(&header[0], hlen);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) return false;
    const size_t a = header.find('('), b = header.find(')');
    if (a == std::string::npos || b == std::string::npos) return false;
    std::stringstream ss(header.substr(a + 1, b - a - 1));
    std::string tok;
    size_t n = 1;
    shape.clear();
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_of("0123456789") == std::string::npos) continue;
        shape.push_back((size_t) std::stoul(tok));
        n *= shape.back();
    }
    data.resize(n);
    f.read((char *) data.data(), (std::streamsize) (n * sizeof(float)));
    return (size_t) f.gcount() == n * sizeof(float);
}

// whatever is available on stdin right now (blocks only until some bytes arrive); 0 at the end
static long read_some(char * buf, size_t n) {
#ifdef _WIN32
    return (long) _read(0, buf, (unsigned) n);
#else
    return (long) read(0, buf, n);
#endif
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    std::string model, voice_path, text, text_file, out = "out.wav", noise_path, bench_path;
    int threads = 4, pause_ms = 0;
    uint64_t seed = 0;
    float speed = 1.0f;
    int lang = 0;
    bool rate_eq = true, post = true, retry = true, one_sentence = false, normalize_only = false, quiet = false;
    bool from_stdin = false, use_gpu = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--voice") voice_path = next();
        else if (a == "-t" || a == "--text") text = next();
        else if (a == "-f" || a == "--file") text_file = next();
        else if (a == "-o" || a == "--out") out = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--speed") speed = (float) std::atof(next().c_str());
        else if (a == "--lang") lang = next() == "en" ? 1 : 0;
        else if (a == "--pause") pause_ms = std::atoi(next().c_str());
        else if (a == "--no-rate-eq") rate_eq = false;
        else if (a == "--no-post") post = false;
        else if (a == "--no-retry") retry = false;
        else if (a == "--stdin") from_stdin = true;
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--sentence") one_sentence = true;
        else if (a == "--noise") { noise_path = next(); one_sentence = true; }
        else if (a == "--normalize") normalize_only = true;
        else if (a == "--bench") bench_path = next();
        else if (a == "-q" || a == "--quiet") quiet = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
    }
    if (!text_file.empty()) text = read_file(text_file);

    if (normalize_only) {
        std::string rest;
        for (const std::string & s : split_sentences(text, true, rest))
            std::printf("%s\n  -> %s\n", s.c_str(), normalize_text(s).c_str());
        return 0;
    }
    if (model.empty() || voice_path.empty() || (text.empty() && bench_path.empty() && !from_stdin)) { usage(); return 2; }

    Engine eng;
    double t0 = now_s();
    if (!eng.load(model, threads, use_gpu)) { std::fprintf(stderr, "error: %s\n", eng.error().c_str()); return 1; }
    const HParams & hp = eng.hp();
    if (!quiet) std::fprintf(stderr, "model %s (%s weights, %d repacked) loaded in %.2f s on %s, %d threads\n", model.c_str(),
                             hp.weight_type.c_str(), eng.repacked_tensors(), now_s() - t0, eng.device_name(), threads);

    std::vector<float> vw;
    int vsr = 0;
    std::string err;
    if (!read_wav(voice_path, vw, vsr, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (vsr != hp.sample_rate) {
        if (!quiet) std::fprintf(stderr, "resampling the voice from %d Hz to %d Hz\n", vsr, hp.sample_rate);
        vw = resample(vw, vsr, hp.sample_rate);
    }
    Voice voice;
    t0 = now_s();
    try {
        if (!eng.make_voice(vw.data(), (int) vw.size(), voice)) { std::fprintf(stderr, "error: %s\n", eng.error().c_str()); return 1; }
    } catch (const std::exception & e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    if (rate_eq && bench_path.empty()) eng.voice_base_rate(voice);
    if (!quiet) std::fprintf(stderr, "voice: %d latent frames, cache built in %.3f s\n", voice.n_ref, now_s() - t0);

    SynthOptions o;
    o.seed = seed;
    o.speed = std::max(0.7f, std::min(1.6f, speed));
    o.language = lang;
    o.equalize_rate = rate_eq;
    o.postprocess = post;
    o.retry_on_artifact = retry;

    if (!bench_path.empty()) {
        // like the original evaluation script: warm-up sentence, then the whole chain per sentence (no rate equalization,
        // no artifact check, low shelf on, seed 42 + i), voice cache computed beforehand
        std::vector<std::string> lines;
        std::stringstream ss(read_file(bench_path));
        std::string line;
        while (std::getline(ss, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        o.equalize_rate = false;
        o.postprocess = true;
        o.retry_on_artifact = false;   // rtf.py has no artifact check either (it costs ~1-2 ms per sentence)
        o.seed = 42;
        eng.synthesize(voice, "Ein kurzer Satz zum Aufwärmen.", o);
        std::vector<double> times;
        double audio_s = 0.0;
        Timing tm;
        for (size_t i = 0; i < lines.size(); i++) {
            o.seed = 42 + i;
            const double a = now_s();
            std::vector<float> w = eng.synthesize(voice, lines[i], o, nullptr, &tm);
            times.push_back(now_s() - a);
            audio_s += (double) w.size() / hp.sample_rate;
        }
        double total = 0.0;
        for (double t : times) total += t;
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted.size() % 2 ? sorted[sorted.size() / 2]
                                             : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
        std::printf("%s%s | %d threads | %.1f s audio in %.2f s = %.1fx real time | first sentence %.3f s, median %.3f s"
                    " | text %.3f dit %.3f decoder %.3f post %.3f\n",
                    hp.weight_type.c_str(), use_gpu ? (std::string(" ") + eng.device_name()).c_str() : "", threads, audio_s, total, audio_s / total, times.empty() ? 0.0 : times[0], med,
                    tm.text, tm.dit, tm.decoder, tm.post);
        if (std::getenv("ARTICULINO_PROFILE")) eng.print_profile();
        return 0;
    }

    std::vector<float> audio;
    Timing tm;
    const bool to_stdout = out == "-";
    size_t n_sentences = 0;
    int lang_state = -1;
    // a finished sentence: raw 16-bit PCM to stdout at once (streaming), or collected for the WAV file
    auto speak = [&](const std::string & sentence) {
        std::string s = sentence;
        if (lang_state == 1) s = "[en]" + s;
        else if (lang_state == 0) s = "[de]" + s;
        lang_state = markup_language_after(sentence, lang_state);
        const double a = now_s();
        std::vector<float> w = eng.synthesize(voice, s, o, nullptr, &tm);
        if (!quiet) std::fprintf(stderr, "[%zu] %.2f s audio in %.3f s: %s\n", n_sentences + 1, (double) w.size() / hp.sample_rate,
                                 now_s() - a, sentence.c_str());
        if (n_sentences > 0 && pause_ms > 0) w.insert(w.begin(), (size_t) hp.sample_rate * pause_ms / 1000, 0.0f);
        n_sentences++;
        if (to_stdout) {
            std::vector<int16_t> pcm(w.size());
            for (size_t i = 0; i < w.size(); i++) pcm[i] = (int16_t) (std::max(-1.0f, std::min(1.0f, w[i])) * 32767.0f);
            std::fwrite(pcm.data(), 2, pcm.size(), stdout);
            std::fflush(stdout);
        }
        audio.insert(audio.end(), w.begin(), w.end());
    };
    t0 = now_s();
    try {
        if (one_sentence) {
            if (!noise_path.empty()) {
                std::vector<float> nz;
                std::vector<size_t> shape;
                if (!read_npy(noise_path, nz, shape) || shape.size() != 3 || shape[0] != 4 || (int) shape[2] != hp.latent) {
                    std::fprintf(stderr, "error: %s must be a float32 .npy of shape (4, frames, %d)\n", noise_path.c_str(), hp.latent);
                    return 1;
                }
                const size_t frames = shape[1];
                o.noise = [nz, frames](int step, int n, int lat, float * dst) {
                    if ((size_t) n != frames) throw std::runtime_error("noise file has " + std::to_string(frames) +
                                                                       " frames, the sentence needs " + std::to_string(n));
                    std::memcpy(dst, &nz[(size_t) step * n * lat], (size_t) n * lat * sizeof(float));
                };
            }
            speak(text);
        } else if (from_stdin) {
            // text arrives piece by piece (e.g. from a language model): speak every sentence as soon as it is complete
            std::string buffer, rest;
            char buf[4096];
            long got;
            while ((got = read_some(buf, sizeof(buf))) > 0) {
                buffer.append(buf, (size_t) got);
                for (const std::string & s : split_sentences(buffer, false, rest)) speak(s);
                buffer = rest;
            }
            for (const std::string & s : split_sentences(buffer, true, rest)) speak(s);
        } else {
            std::string rest;
            for (const std::string & s : split_sentences(text, true, rest)) speak(s);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const double dt = now_s() - t0;
    if (!to_stdout && !write_wav(out, audio.data(), audio.size(), hp.sample_rate)) {
        std::fprintf(stderr, "error: cannot write %s\n", out.c_str());
        return 1;
    }
    const double secs = (double) audio.size() / hp.sample_rate;
    if (!quiet) std::fprintf(stderr, "%s: %.2f s audio in %.3f s (%.1fx real time) | text %.3f dit %.3f decoder %.3f post %.3f%s\n",
                             to_stdout ? "stdout" : out.c_str(), secs, dt, dt > 0 ? secs / dt : 0.0, tm.text, tm.dit, tm.decoder,
                             tm.post, tm.retries ? (" | " + std::to_string(tm.retries) + " sentence(s) remade").c_str() : "");
    return 0;
}
