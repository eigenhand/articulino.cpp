// Audio front and back end: log-mel spectrogram, inverse STFT, WAV files, resampling, low shelf filter.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <thread>

namespace articulino {

// run fn(begin, end) over [0, n) on up to n_threads threads
template <typename F>
static void parallel_for(int n, int n_threads, F fn) {
    n_threads = std::max(1, std::min(n_threads, n / 32));
    if (n_threads <= 1) { fn(0, n); return; }
    std::vector<std::thread> th;
    const int chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        const int b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        th.emplace_back(fn, b, e);
    }
    for (auto & t : th) t.join();
}

// torchaudio MelSpectrogram(24 kHz, n_fft 1024, win 960, hop 240, 100 mels, power 1, center, reflect) + log(clamp 1e-5)
std::vector<float> Engine::logmel(const float * wav, int n, int & frames) const {
    const int nfft = hp_.mel_nfft, hop = hp_.mel_hop, win = hp_.mel_win, nm = hp_.n_mels;
    const int nf = nfft / 2 + 1, pad = nfft / 2;
    if (n <= pad) { frames = 0; return {}; }
    frames = n / hop + 1;
    std::vector<float> window(nfft, 0.0f);
    const int off = (nfft - win) / 2;
    for (int i = 0; i < win; i++) window[off + i] = hann_[i];
    std::vector<float> out((size_t) frames * nm);
    parallel_for(frames, n_threads_, [&](int b, int e) {
        std::vector<float> buf(nfft);
        std::vector<std::complex<float>> spec(nf);
        std::vector<float> mag(nf);
        const pocketfft::shape_t shape{ (size_t) nfft };
        const pocketfft::stride_t sin{ sizeof(float) }, sout{ sizeof(std::complex<float>) };
        for (int f = b; f < e; f++) {
            for (int j = 0; j < nfft; j++) {
                int idx = f * hop + j - pad;                               // reflect padding
                if (idx < 0) idx = -idx;
                if (idx >= n) idx = 2 * (n - 1) - idx;
                buf[j] = wav[idx] * window[j];
            }
            pocketfft::r2c(shape, sin, sout, 0, pocketfft::FORWARD, buf.data(), spec.data(), 1.0f);
            for (int k = 0; k < nf; k++) mag[k] = std::abs(spec[k]);
            float * o = &out[(size_t) f * nm];
            for (int m = 0; m < nm; m++) o[m] = 0.0f;
            for (int k = 0; k < nf; k++) {
                const float a = mag[k];
                const float * fb = &mel_fb_[(size_t) k * nm];
                for (int m = 0; m < nm; m++) o[m] += fb[m] * a;
            }
            for (int m = 0; m < nm; m++) o[m] = std::log(std::max(o[m], 1e-5f));
        }
    });
    return out;
}

// torch.istft(n_fft 960, hop 240, Hann 960, center, length) of mag = clamp(exp(logmag), 100), phase
std::vector<float> Engine::istft(const std::vector<float> & head, int n_frames, int length) const {
    const int nfft = hp_.istft_nfft, hop = hp_.istft_hop, nb = nfft / 2 + 1;
    const int full = nfft + hop * (n_frames - 1);
    std::vector<float> frames((size_t) n_frames * nfft);
    parallel_for(n_frames, n_threads_, [&](int b, int e) {
        std::vector<std::complex<float>> spec(nb);
        std::vector<float> buf(nfft);
        const pocketfft::shape_t shape{ (size_t) nfft };
        const pocketfft::stride_t sin{ sizeof(std::complex<float>) }, sout{ sizeof(float) };
        for (int f = b; f < e; f++) {
            const float * h = &head[(size_t) f * 2 * nb];
            for (int k = 0; k < nb; k++) {
                const float mag = std::min(std::exp(h[k]), 100.0f);
                const float ph = h[nb + k];
                spec[k] = std::complex<float>(mag * std::cos(ph), mag * std::sin(ph));
            }
            pocketfft::c2r(shape, sin, sout, 0, pocketfft::BACKWARD, spec.data(), buf.data(), 1.0f / (float) nfft);
            float * o = &frames[(size_t) f * nfft];
            for (int j = 0; j < nfft; j++) o[j] = buf[j] * hann_[j];
        }
    });
    std::vector<float> ola(full, 0.0f), env(full, 0.0f);
    for (int f = 0; f < n_frames; f++) {
        const float * o = &frames[(size_t) f * nfft];
        float * a = &ola[(size_t) f * hop];
        float * en = &env[(size_t) f * hop];
        for (int j = 0; j < nfft; j++) {
            a[j] += o[j];
            en[j] += hann_[j] * hann_[j];
        }
    }
    std::vector<float> out(length, 0.0f);
    const int start = nfft / 2;
    for (int i = 0; i < length; i++) {
        const int j = start + i;
        if (j >= full) break;
        out[i] = env[j] > 1e-11f ? ola[j] / env[j] : 0.0f;
    }
    return out;
}

// server.auffaellig(), tone part: per frame (n 1024, hop 256, symmetric Hann) the energy of the strongest bin above
// 1.8 kHz +- 2 bins; more than 35 % of the frame energy for >= 60 ms in a row counts as a whistle
bool narrow_band_tone(const float * x, size_t len, int sr, float & ms, float & hz, int n_threads) {
    const int n = 1024, h = 256, nb = n / 2 + 1;
    ms = hz = 0;
    if (len < (size_t) 2 * n) return false;
    const int k = (int) ((len - n) / h + 1);
    std::vector<float> win(n);
    for (int i = 0; i < n; i++) win[i] = (float) (0.5 - 0.5 * std::cos(2.0 * M_PI * i / (n - 1)));
    const int ab = (int) (1800.0 / ((double) sr / n));
    std::vector<int> peak(k, -1);   // strongest bin of a frame that counts as a hit, -1 = no hit
    parallel_for(k, n_threads, [&](int b, int e) {
        std::vector<float> buf(n);
        std::vector<std::complex<float>> spec(nb);
        std::vector<double> sp(nb);
        const pocketfft::shape_t shape{ (size_t) n };
        const pocketfft::stride_t sin{ sizeof(float) }, sout{ sizeof(std::complex<float>) };
        for (int f = b; f < e; f++) {
            for (int i = 0; i < n; i++) buf[i] = x[(size_t) f * h + i] * win[i];
            pocketfft::r2c(shape, sin, sout, 0, pocketfft::FORWARD, buf.data(), spec.data(), 1.0f);
            double total = 1e-12;
            for (int q = 0; q < nb; q++) { sp[q] = std::norm(spec[q]); total += sp[q]; }
            int g = ab;
            for (int q = ab; q < nb; q++) if (sp[q] > sp[g]) g = q;
            double narrow = 0;
            for (int q = std::max(ab, g - 2); q < std::min(nb, g + 3); q++) narrow += sp[q];
            if (narrow / total > 0.35 && total > 1e-2 * n) peak[f] = g;
        }
    });
    int run = 0, best = 0;
    std::vector<int> hits;
    for (int f = 0; f < k; f++) {
        run = peak[f] >= 0 ? run + 1 : 0;
        best = std::max(best, run);
        if (peak[f] >= 0) hits.push_back(peak[f]);
    }
    ms = (float) (best * h * 1000.0 / sr);
    if (!hits.empty()) { std::sort(hits.begin(), hits.end()); hz = (float) (hits[hits.size() / 2] * (double) sr / n); }
    return ms >= 60.0f;
}

// ------------------------------------------------------------------------------------------------ WAV

static uint32_t rd32(const uint8_t * p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24); }
static uint16_t rd16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }

bool read_wav(const std::string & path, std::vector<float> & mono, int & sample_rate, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> d;
    uint8_t buf[65536];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + r);
    std::fclose(f);
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) != 0 || std::memcmp(d.data() + 8, "WAVE", 4) != 0) {
        err = path + ": not a RIFF/WAVE file"; return false;
    }
    int fmt = 0, ch = 0, bits = 0;
    sample_rate = 0;
    const uint8_t * data = nullptr;
    size_t data_len = 0;
    size_t p = 12;
    while (p + 8 <= d.size()) {
        const uint32_t len = rd32(&d[p + 4]);
        const uint8_t * body = &d[p + 8];
        const size_t avail = std::min<size_t>(len, d.size() - p - 8);
        if (std::memcmp(&d[p], "fmt ", 4) == 0 && avail >= 16) {
            fmt = rd16(body); ch = rd16(body + 2); sample_rate = (int) rd32(body + 4); bits = rd16(body + 14);
            if (fmt == 0xFFFE && avail >= 26) fmt = rd16(body + 24);     // WAVE_FORMAT_EXTENSIBLE: sub format
        } else if (std::memcmp(&d[p], "data", 4) == 0) {
            data = body; data_len = avail;
        }
        p += 8 + len + (len & 1);
    }
    if (!data || ch <= 0 || sample_rate <= 0) { err = path + ": missing fmt or data chunk"; return false; }
    const int bps = bits / 8;
    if (!((fmt == 1 && (bits == 16 || bits == 24 || bits == 32 || bits == 8)) || (fmt == 3 && bits == 32))) {
        err = path + ": unsupported WAV encoding (use 8/16/24/32-bit PCM or 32-bit float)"; return false;
    }
    const size_t n = data_len / (size_t) (bps * ch);
    mono.assign(n, 0.0f);
    for (size_t i = 0; i < n; i++) {
        double acc = 0.0;
        for (int c = 0; c < ch; c++) {
            const uint8_t * s = data + (i * ch + c) * bps;
            double v = 0.0;
            if (fmt == 3) { float x; std::memcpy(&x, s, 4); v = x; }
            else if (bits == 8) v = ((int) s[0] - 128) / 128.0;
            else if (bits == 16) v = (int16_t) rd16(s) / 32768.0;
            else if (bits == 24) v = (int32_t) ((uint32_t) (s[0] << 8 | s[1] << 16 | (uint32_t) s[2] << 24)) / 2147483648.0;
            else v = (int32_t) rd32(s) / 2147483648.0;
            acc += v;
        }
        mono[i] = (float) (acc / ch);
    }
    return true;
}

bool write_wav(const std::string & path, const float * x, size_t n, int sample_rate) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { uint8_t b[4] = { (uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24) }; std::fwrite(b, 1, 4, f); };
    auto w16 = [&](uint16_t v) { uint8_t b[2] = { (uint8_t) v, (uint8_t) (v >> 8) }; std::fwrite(b, 1, 2, f); };
    std::fwrite("RIFF", 1, 4, f); w32((uint32_t) (36 + n * 2)); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32((uint32_t) sample_rate); w32((uint32_t) sample_rate * 2); w16(2); w16(16);
    std::fwrite("data", 1, 4, f); w32((uint32_t) (n * 2));
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; i++) {
        const float v = std::max(-1.0f, std::min(1.0f, x[i]));
        pcm[i] = (int16_t) (v * 32767.0f);   // same truncation as the reference service ((wav * 32767).astype(int16))
    }
    const bool ok = std::fwrite(pcm.data(), 2, n, f) == n;
    std::fclose(f);
    return ok;
}

// ------------------------------------------------------------------------------------------------ resampling

// windowed-sinc resampler (Kaiser window, beta 8.6, 32 zero crossings), for voices that are not 24 kHz
std::vector<float> resample(const std::vector<float> & x, int sr_in, int sr_out) {
    if (sr_in == sr_out || x.empty()) return x;
    const double ratio = (double) sr_out / sr_in;
    const double cutoff = std::min(1.0, ratio) * 0.97;
    const int zc = 32;
    const double beta = 8.6;
    auto bessel_i0 = [](double v) {
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 50; k++) { t *= (v / (2.0 * k)) * (v / (2.0 * k)); s += t; }
        return s;
    };
    const double i0b = bessel_i0(beta);
    const size_t n_out = (size_t) std::floor(x.size() * ratio);
    std::vector<float> y(n_out);
    const double half = zc / cutoff;
    for (size_t i = 0; i < n_out; i++) {
        const double t = i / ratio;
        const long lo = (long) std::ceil(t - half), hi = (long) std::floor(t + half);
        double acc = 0.0;
        for (long j = lo; j <= hi; j++) {
            if (j < 0 || j >= (long) x.size()) continue;
            const double d = (t - j) * cutoff;
            const double sinc = d == 0.0 ? 1.0 : std::sin(M_PI * d) / (M_PI * d);
            const double r = (t - j) / half;
            const double win = std::fabs(r) >= 1.0 ? 0.0 : bessel_i0(beta * std::sqrt(1.0 - r * r)) / i0b;
            acc += x[j] * sinc * win;
        }
        y[i] = (float) (acc * cutoff);
    }
    return y;
}

// RBJ cookbook low shelf (as server.py `_shelf`), applied like scipy sosfilt in double precision
void low_shelf(std::vector<float> & x, float f0, float db, int sr, float slope) {
    if (db == 0.0f || x.empty()) return;
    const double A = std::pow(10.0, db / 40.0), w = 2.0 * M_PI * f0 / sr, c = std::cos(w);
    const double q = 2.0 * std::sqrt(A) * std::sin(w) / 2.0 * std::sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0);
    const double z = -1.0;
    double b0 = A * ((A + 1) + z * (A - 1) * c + q), b1 = -2 * z * A * ((A - 1) + z * (A + 1) * c),
           b2 = A * ((A + 1) + z * (A - 1) * c - q);
    double a0 = (A + 1) - z * (A - 1) * c + q, a1 = 2 * z * ((A - 1) - z * (A + 1) * c), a2 = (A + 1) - z * (A - 1) * c - q;
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
    double z1 = 0.0, z2 = 0.0;
    for (float & s : x) {
        const double in = s;
        const double out = b0 * in + z1;
        z1 = b1 * in - a1 * out + z2;
        z2 = b2 * in - a2 * out;
        s = (float) out;
    }
}

}  // namespace articulino
