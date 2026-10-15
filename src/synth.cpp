// Voice creation and the per-sentence pipeline.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace articulino {

bool Engine::make_voice(const float * wav, int n, Voice & v) {
    const int hop = hp_.samples_per_latent;
    const int n_use = std::min(n, hp_.voice_max_samples) / hop * hop;
    if (n_use < 2 * hop) { err_ = "voice recording too short"; return false; }
    int frames = 0;
    std::vector<float> mel = logmel(wav, n_use, frames);
    const int use = n_use / hp_.mel_hop;          // exactly T/240 frames (the transform gives one more)
    mel.resize((size_t) use * hp_.n_mels);
    int n_lat = 0;
    std::vector<float> mu = encode_latents(mel, use, n_lat);
    v.n_ref = n_lat;
    v.ref.resize(mu.size());
    for (int t = 0; t < n_lat; t++)
        for (int d = 0; d < hp_.latent; d++)
            v.ref[(size_t) t * hp_.latent + d] = (mu[(size_t) t * hp_.latent + d] - hp_.latent_mean[d]) / hp_.latent_std[d];
    voice_encode(v.ref, v.n_ref, v.tokens, v.vec);
    v.base_rate = -1.0f;
    return build_prompt_cache(v);
}

// abtasten.buchstaben_rate: letters per second over the letters from position `from` on; with target > 0 the
// letter durations are scaled in place to exactly that rate
static float letter_rate(std::vector<float> & d, const std::vector<int32_t> & ids, size_t from, float target) {
    std::vector<size_t> idx;
    for (size_t i = from; i < ids.size(); i++)
        if (is_alnum_symbol(ids[i])) idx.push_back(i);
    if (idx.empty()) return 0.0f;
    std::vector<float> sel(idx.size());
    for (size_t i = 0; i < idx.size(); i++) sel[i] = std::max(d[idx[i]], 0.5f);
    std::vector<float> s = sel;
    std::sort(s.begin(), s.end());
    const size_t m = s.size();
    const float med = (m % 2) ? s[m / 2] : (s[m / 2 - 1] + s[m / 2]) / 2.0f;
    float sum = 0.0f;
    for (float & x : sel) { x = std::min(x, 3.0f * med); sum += x; }
    double rate = (double) m / (sum / 100.0f);
    if (target > 0.0f) {
        const double f = rate / target;
        for (size_t i = 0; i < idx.size(); i++) d[idx[i]] = (float) (sel[i] * f);
        rate = target;
    }
    return (float) rate;
}

float Engine::voice_base_rate(Voice & v) {
    if (v.base_rate > 0.0f) return v.base_rate;
    std::vector<float> rates;
    for (const std::string & s : hp_.rate_calibration) {
        std::vector<int32_t> ids, lang;
        text_to_ids(s, 0, ids, lang);
        if (ids.empty()) continue;
        std::vector<float> mask(ids.size(), 1.0f), h, dl;
        text_and_durations(ids, lang, mask, v, h, dl);
        std::vector<float> d(dl.size());
        for (size_t i = 0; i < d.size(); i++) d[i] = std::exp(dl[i]);
        rates.push_back(letter_rate(d, ids, 0, 0.0f));
    }
    if (rates.empty()) { v.base_rate = 14.0f; return v.base_rate; }
    std::sort(rates.begin(), rates.end());
    const size_t m = rates.size();
    v.base_rate = (m % 2) ? rates[m / 2] : (float) (((double) rates[m / 2 - 1] + rates[m / 2]) / 2.0);
    return v.base_rate;
}

std::vector<float> Engine::durations(const std::vector<float> & dur_log, const std::vector<float> & mask,
                                     const std::vector<int32_t> & ids, const Voice & v, const SynthOptions & o) {
    std::vector<float> d(dur_log.size());
    for (size_t i = 0; i < d.size(); i++) d[i] = std::min(std::exp(dur_log[i]) * mask[i], hp_.duration_max) / o.speed;
    if (o.equalize_rate) letter_rate(d, ids, 1, voice_base_rate(const_cast<Voice &>(v)) * o.speed);
    d[0] = (float) (hp_.units_per_latent * v.n_prompt);
    return d;
}

// modell.hochtasten: weight of letter l on frame j = overlap of [start_l, end_l) with [4j, 4j+4), divided by 4
std::vector<float> Engine::upsample(const std::vector<float> & h, const std::vector<float> & dur, int n_frames) const {
    const int C = hp_.text_dim, U = hp_.units_per_latent;
    const int N = (int) dur.size();
    std::vector<float> out((size_t) n_frames * C, 0.0f);
    double acc = 0.0;
    for (int l = 0; l < N; l++) {
        const float d = dur[l];
        acc += d;
        const float end = (float) acc;          // torch cumsum accumulates in double on the CPU
        const float start = end - d;
        if (!(end > start)) continue;
        int j0 = std::max(0, (int) std::floor(start / U));
        int j1 = std::min(n_frames - 1, (int) std::ceil(end / U));
        const float * hl = &h[(size_t) l * C];
        for (int j = j0; j <= j1; j++) {
            const float lo = (float) (U * j);
            const float w = std::max(std::min(end, lo + U) - std::max(start, lo), 0.0f) / (float) U;
            if (w <= 0.0f) continue;
            float * o = &out[(size_t) j * C];
            for (int c = 0; c < C; c++) o[c] += w * hl[c];
        }
    }
    return out;
}

// server.auffaellig(): latent outliers, or a narrow-band tone above 1.8 kHz for at least 60 ms ('' = fine)
std::string Engine::artifact(const std::vector<float> & z, const float * wav, size_t n) const {
    double sum = 0, sq = 0, peak = 0;
    for (float x : z) { sum += x; sq += (double) x * x; peak = std::max(peak, (double) std::fabs(x)); }
    const double m = z.empty() ? 0 : sum / z.size();
    const double sd = z.size() > 1 ? std::sqrt(std::max(0.0, (sq - z.size() * m * m) / (z.size() - 1))) : 0.0;
    if (!std::isfinite(sd) || sd > 1.6 || peak > 9.0) return "latent outliers";
    float ms = 0, hz = 0;
    if (narrow_band_tone(wav, n, hp_.sample_rate, ms, hz, n_threads_)) return "tone";
    return "";
}

std::vector<float> Engine::synthesize(Voice & v, const std::string & sentence, const SynthOptions & o,
                                      SynthTrace * trace, Timing * timing) {
    const double t0 = now_s();
    std::vector<int32_t> raw, raw_lang;
    text_to_ids(sentence, o.language, raw, raw_lang);
    if (raw.empty()) return {};
    std::vector<int32_t> ids{ 0 }, lang{ raw_lang[0] };
    ids.insert(ids.end(), raw.begin(), raw.end());
    lang.insert(lang.end(), raw_lang.begin(), raw_lang.end());
    std::vector<float> mask(ids.size(), 1.0f);
    mask[0] = 0.0f;
    std::vector<float> h, dl;
    text_and_durations(ids, lang, mask, v, h, dl);
    std::vector<float> d = durations(dl, mask, ids, v, o);
    double total = 0.0;
    for (float x : d) total += x;
    const int L = std::max(1, (int) std::ceil(total / hp_.units_per_latent));
    const int P = v.n_prompt;
    const int Ln = L - P;
    if (Ln <= 0) return {};
    std::vector<float> hf = upsample(h, d, L);
    const double t1 = now_s();

    // sampling + decoding; like the reference service, a sentence that comes out broken (latent outliers or a
    // narrow-band whistle) is made once more with the next seed
    const int ctx = std::min(hp_.decoder_context, P);
    const int lat = hp_.latent;
    const int T = ctx + Ln;
    std::vector<float> hf_new(hf.begin() + (size_t) P * hp_.text_dim, hf.end());
    std::vector<float> x1, wav;
    std::string why;
    double dit_s = 0, dec_s = 0, check_s = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
        NoiseFn noise = o.noise;
        Rng rng(o.seed + (uint64_t) attempt);
        if (!noise) noise = [&rng](int, int n, int l, float * out) { rng.normal(out, (size_t) n * l); };
        const double a0 = now_s();
        x1 = sample(v, hf_new, Ln, noise, trace);
        const double a1 = now_s();
        dit_s += a1 - a0;
        std::vector<float> z((size_t) T * lat);
        for (int t = 0; t < T; t++) {
            const float * src = t < ctx ? &v.ref[(size_t) (P - ctx + t) * lat] : &x1[(size_t) (t - ctx) * lat];
            for (int c = 0; c < lat; c++) z[(size_t) t * lat + c] = src[c] * hp_.latent_std[c] + hp_.latent_mean[c];
        }
        wav = decode(z, T);
        const double a2 = now_s();
        dec_s += a2 - a1;
        if (!o.retry_on_artifact || o.noise || attempt == 1) break;
        why = artifact(x1, wav.data() + (size_t) ctx * hp_.samples_per_latent, wav.size() - (size_t) ctx * hp_.samples_per_latent);
        check_s += now_s() - a2;
        if (why.empty()) break;
        if (timing) timing->retries++;
    }
    const double t3 = now_s();

    if (trace) {
        trace->ids = ids;
        trace->text_h = h;
        trace->dur_log = dl;
        trace->durations = d;
        trace->frame_cond = hf;
        trace->frames_total = L;
        trace->frames_new = Ln;
        trace->latents_norm.assign(v.ref.begin(), v.ref.begin() + (size_t) P * lat);
        trace->latents_norm.insert(trace->latents_norm.end(), x1.begin(), x1.end());
        trace->decoder_out = wav;
        trace->artifact = why;
        trace->audio.assign(wav.begin() + (size_t) ctx * hp_.samples_per_latent, wav.end());
    }
    if (o.postprocess) low_shelf(wav, hp_.shelf_hz, hp_.shelf_db, hp_.sample_rate, hp_.shelf_slope);
    std::vector<float> out(wav.begin() + (size_t) ctx * hp_.samples_per_latent, wav.end());
    if (o.postprocess) {
        float peak = 0.0f;
        for (float s : out) peak = std::max(peak, std::fabs(s));
        if (peak > hp_.peak) for (float & s : out) s *= hp_.peak / peak;
    }
    const double t4 = now_s();
    if (timing) {
        timing->text += t1 - t0;
        timing->dit += dit_s;
        timing->decoder += dec_s;
        timing->post += t4 - t3 + check_s;
        timing->total += t4 - t0;
    }
    return out;
}

}  // namespace articulino
