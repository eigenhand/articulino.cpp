// Neural network parts as ggml graphs. Activations are laid out as [channels, time] (ne0 = channels).
#include "engine.h"
#include "ops_cpu.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace articulino {

namespace {

ggml_tensor * linear(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, y, b) : y;
}

ggml_tensor * layer_norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * n = ggml_norm(c, x, eps);
    if (w) n = ggml_mul(c, n, w);
    if (b) n = ggml_add(c, n, b);
    return n;
}

// part p of a fused projection [n_parts * D, L] as [dh, H, L] (strided view)
ggml_tensor * part(ggml_context * c, ggml_tensor * fused, int p, int64_t dh, int64_t H) {
    const int64_t L = fused->ne[1];
    const size_t es = ggml_element_size(fused);
    return ggml_view_3d(c, fused, dh, H, L, dh * es, fused->nb[1], (size_t) p * dh * H * es);
}

// attention on prepared operands: qp [dh, Lq, H] contiguous, kp [dh, Lk, H], vt [Lk, dh, H] contiguous -> [D, Lq]
ggml_tensor * attention_core(ggml_context * c, ggml_tensor * qp, ggml_tensor * kp, ggml_tensor * vt, ggml_tensor * mask) {
    const int64_t dh = qp->ne[0], lq = qp->ne[1], H = qp->ne[2];
    ggml_tensor * kq = ggml_mul_mat(c, kp, qp);                          // [Lk, Lq, H]
    kq = ggml_soft_max_ext(c, kq, mask, 1.0f / std::sqrt((float) dh), 0.0f);
    ggml_tensor * o = ggml_mul_mat(c, vt, kq);                           // [dh, Lq, H]
    o = ggml_permute(c, o, 0, 2, 1, 3);                                  // [dh, H, Lq]
    return ggml_cont_2d(c, o, dh * H, lq);
}

ggml_tensor * q_layout(ggml_context * c, ggml_tensor * q) { return ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)); }   // [dh, L, H]
ggml_tensor * k_layout(ggml_context * c, ggml_tensor * k) { return ggml_permute(c, k, 0, 2, 1, 3); }                 // [dh, L, H]
ggml_tensor * v_layout(ggml_context * c, ggml_tensor * v) { return ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)); }   // [L, dh, H]

// q/k/v [dh, H, L] (any strides) -> [D, Lq]; mask [Lk, Lq] additive or null
ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) {
    return attention_core(c, q_layout(c, q), k_layout(c, k), v_layout(c, v), mask);
}

// x [C, T] -> [K*C, T]: column t holds x[:, t+k-pad] for k = 0..K-1 (zero padded), index k*C + c
ggml_tensor * unfold(ggml_context * c, ggml_tensor * x, int K, int pad) {
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * xp = ggml_pad_ext(c, x, 0, 0, pad, K - 1 - pad, 0, 0, 0, 0);
    ggml_tensor * out = nullptr;
    for (int k = 0; k < K; k++) {
        ggml_tensor * v = ggml_view_2d(c, xp, C, T, xp->nb[1], (size_t) k * xp->nb[1]);
        out = out ? ggml_concat(c, out, v, 0) : ggml_cont(c, v);
    }
    return out;
}

// AdaNorm: LayerNorm without weights, then x * scale1p + shift (scale1p = 1 + scale)
ggml_tensor * modulate(ggml_context * c, ggml_tensor * x, ggml_tensor * scale1p, ggml_tensor * shift) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), scale1p), shift);
}

// split an AdaNorm projection [2D, 1] into (1 + scale, shift)
void ada_split(ggml_context * c, ggml_tensor * ada, int64_t D, ggml_tensor ** scale1p, ggml_tensor ** shift) {
    *scale1p = ggml_scale_bias(c, ggml_view_1d(c, ada, D, 0), 1.0f, 1.0f);
    *shift = ggml_view_1d(c, ada, D, (size_t) D * ggml_element_size(ada));
}

// time feature of the DiT: concat(sin(w), cos(w)), w = t * 1000 * exp(-ln(10000) * i / 128)
std::vector<float> time_feature(float t) {
    std::vector<float> f(256);
    for (int i = 0; i < 128; i++) {
        const float fr = std::exp(-std::log(10000.0f) * (float) i / 128.0f);
        const float w = t * 1000.0f * fr;
        f[i] = std::sin(w);
        f[128 + i] = std::cos(w);
    }
    return f;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ shared blocks

ggml_tensor * Engine::gelu(ggml_context * c, ggml_tensor * x) const {
    return fused_ ? gelu_fast(c, x) : ggml_gelu_erf(c, x);
}

ggml_tensor * Engine::rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos) const {
    return ggml_rope_ext(c, x, pos, nullptr, (int) x->ne[0], GGML_ROPE_TYPE_NORMAL, 0, hp_.rope_base, 1.0f, 0.0f, 1.0f,
                         0.0f, 0.0f);
}

ggml_tensor * Engine::tblock(Graph & g, const std::string & p, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask,
                             int heads) {
    ggml_context * c = g.ctx;
    const int64_t D = x->ne[0], dh = D / heads;
    ggml_tensor * h = layer_norm(c, x, W(p + ".attn_norm.weight"), W(p + ".attn_norm.bias"), 1e-5f);
    ggml_tensor * qkv = linear(c, h, W(p + ".attn_qkv.weight"), W(p + ".attn_qkv.bias"));
    ggml_tensor * q = part(c, qkv, 0, dh, heads), * k = part(c, qkv, 1, dh, heads), * v = part(c, qkv, 2, dh, heads);
    if (pos) { q = rope(c, q, pos); k = rope(c, k, pos); }
    x = ggml_add(c, x, linear(c, attention(c, q, k, v, mask), W(p + ".attn_out.weight"), W(p + ".attn_out.bias")));
    h = layer_norm(c, x, W(p + ".ffn_norm.weight"), W(p + ".ffn_norm.bias"), 1e-5f);
    h = gelu(c, linear(c, h, W(p + ".ffn_up.weight"), W(p + ".ffn_up.bias")));
    return ggml_add(c, x, linear(c, h, W(p + ".ffn_down.weight"), W(p + ".ffn_down.bias")));
}

ggml_tensor * Engine::convnext(Graph & g, const std::string & p, ggml_tensor * x) {
    ggml_context * c = g.ctx;
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * dw = W(p + ".dw.weight");                              // [C, K]
    const int K = (int) dw->ne[1], pad = K / 2;
    ggml_tensor * acc;
    if (fused_) {
        acc = dwconv_fused(c, x, dw, W(p + ".dw.bias"));
    } else {
        ggml_tensor * xp = ggml_pad_ext(c, x, 0, 0, pad, pad, 0, 0, 0, 0);    // [C, T + 2 pad]
        acc = nullptr;
        for (int k = 0; k < K; k++) {
            ggml_tensor * seg = ggml_view_2d(c, xp, C, T, xp->nb[1], (size_t) k * xp->nb[1]);
            ggml_tensor * wk = ggml_view_1d(c, dw, C, (size_t) k * dw->nb[1]);
            ggml_tensor * t = ggml_mul(c, seg, wk);
            acc = acc ? ggml_add(c, acc, t) : t;
        }
        acc = ggml_add(c, acc, W(p + ".dw.bias"));
    }
    ggml_tensor * h = layer_norm(c, acc, W(p + ".norm.weight"), W(p + ".norm.bias"), 1e-6f);
    h = gelu(c, linear(c, h, W(p + ".up.weight"), W(p + ".up.bias")));
    h = linear(c, h, W(p + ".down.weight"), W(p + ".down.bias"));
    return ggml_add(c, x, ggml_mul(c, h, W(p + ".gamma")));
}

// One DiT block. Self-attention keys/values of the prompt in front: kp_prev [dh, P, H], vt_prev [P, dh, H] (null for
// the prompt itself); the block's own keys/values come back through kp_out/vt_out when asked for.
// Cross-attention on the voice tokens: xkp [dh, NT, H], xvt [NT, dh, H].
ggml_tensor * Engine::dit_block(Graph & g, int i, ggml_tensor * x, const DitMod & m, ggml_tensor * pos,
                                ggml_tensor * kp_prev, ggml_tensor * vt_prev, ggml_tensor * xkp, ggml_tensor * xvt,
                                ggml_tensor ** kp_out, ggml_tensor ** vt_out) {
    ggml_context * c = g.ctx;
    const std::string p = "dit.blk." + std::to_string(i);
    const int64_t D = x->ne[0], H = hp_.dit_heads, dh = D / H, L = x->ne[1];
    ggml_tensor * xn = modulate(c, x, m.attn_scale1p, m.attn_shift);
    ggml_tensor * qkv = linear(c, xn, W(p + ".attn_qkv.weight"), W(p + ".attn_qkv.bias"));
    ggml_tensor * q = rope(c, part(c, qkv, 0, dh, H), pos);
    ggml_tensor * kp = k_layout(c, rope(c, part(c, qkv, 1, dh, H), pos));
    ggml_tensor * vt = v_layout(c, part(c, qkv, 2, dh, H));
    if (kp_out) { kp = ggml_cont(c, kp); *kp_out = kp; }
    if (vt_out) *vt_out = vt;
    if (kp_prev) {
        kp = ggml_concat(c, kp_prev, kp, 1);                             // [dh, P + L, H]
        vt = ggml_concat(c, vt_prev, vt, 0);                             // [P + L, dh, H]
    }
    x = ggml_add(c, x, linear(c, attention_core(c, q_layout(c, q), kp, vt, nullptr), W(p + ".attn_out.weight"),
                              W(p + ".attn_out.bias")));
    ggml_tensor * h = layer_norm(c, x, W(p + ".xattn_norm.weight"), W(p + ".xattn_norm.bias"), 1e-5f);
    ggml_tensor * q2 = ggml_reshape_3d(c, linear(c, h, W(p + ".xattn_q.weight"), W(p + ".xattn_q.bias")), dh, H, L);
    x = ggml_add(c, x, linear(c, attention_core(c, q_layout(c, q2), xkp, xvt, nullptr), W(p + ".xattn_out.weight"),
                              W(p + ".xattn_out.bias")));
    h = modulate(c, x, m.ffn_scale1p, m.ffn_shift);
    h = gelu(c, linear(c, h, W(p + ".ffn_up.weight"), W(p + ".ffn_up.bias")));
    return ggml_add(c, x, linear(c, h, W(p + ".ffn_down.weight"), W(p + ".ffn_down.bias")));
}

ggml_tensor * Engine::time_cond(Graph & g, float t, ggml_tensor * g_dit) {
    ggml_context * c = g.ctx;
    const std::vector<float> tf = time_feature(t);
    ggml_tensor * h = ggml_silu(c, linear(c, g.in_f32(tf.data(), 256), W("dit.time_fc1.weight"), W("dit.time_fc1.bias")));
    return ggml_add(c, linear(c, h, W("dit.time_fc2.weight"), W("dit.time_fc2.bias")), g_dit);
}

// ------------------------------------------------------------------------------------------------ latent encoder

std::vector<float> Engine::encode_latents(const std::vector<float> & mel, int frames, int & n_lat) {
    if (frames % 4 != 0 || frames <= 0) throw std::runtime_error("encoder input must be a multiple of 4 mel frames");
    Graph g(8192);
    ggml_context * c = g.ctx;
    ggml_tensor * x = g.in_f32(mel.data(), hp_.n_mels, frames);
    x = linear(c, unfold(c, x, 7, 3), W("enc.in.weight"), W("enc.in.bias"));       // [512, T]
    const int per_stage = hp_.enc_blocks / 3;
    int blk = 0;
    for (int stage = 0; stage < 3; stage++) {
        for (int j = 0; j < per_stage; j++) x = convnext(g, "enc.blk." + std::to_string(blk++), x);
        if (stage < 2) {   // Conv1d k4 s2 p1 as one matmul over frame pairs
            const int64_t C = x->ne[0], T = x->ne[1];
            ggml_tensor * xp = ggml_pad_ext(c, x, 0, 0, 1, 1, 0, 0, 0, 0);          // [C, T+2]
            xp = ggml_reshape_2d(c, xp, 2 * C, (T + 2) / 2);                      // frame pairs
            ggml_tensor * a = ggml_view_2d(c, xp, 2 * C, T / 2, xp->nb[1], 0);
            ggml_tensor * b = ggml_view_2d(c, xp, 2 * C, T / 2, xp->nb[1], xp->nb[1]);
            ggml_tensor * u = ggml_concat(c, ggml_cont(c, a), b, 0);              // [4C, T/2]
            const std::string p = "enc.down." + std::to_string(stage);
            x = linear(c, u, W(p + ".weight"), W(p + ".bias"));
        }
    }
    x = layer_norm(c, x, W("enc.out_norm.weight"), W("enc.out_norm.bias"), 1e-6f);
    x = linear(c, x, W("enc.mu.weight"), W("enc.mu.bias"));                         // [96, T/4]
    g.out(x);
    compute(g);
    n_lat = (int) x->ne[1];
    return get(x);
}

// ------------------------------------------------------------------------------------------------ voice encoder

void Engine::voice_encode(const std::vector<float> & ref, int n_ref, std::vector<float> & tokens, std::vector<float> & vec) {
    Graph g(4096);
    ggml_context * c = g.ctx;
    std::vector<int32_t> pos(n_ref);
    for (int i = 0; i < n_ref; i++) pos[i] = i;
    ggml_tensor * tpos = g.in_i32(pos.data(), n_ref);
    ggml_tensor * x = linear(c, g.in_f32(ref.data(), hp_.latent, n_ref), W("voice.in.weight"), W("voice.in.bias"));
    for (int i = 0; i < hp_.voice_layers; i++) x = tblock(g, "voice.blk." + std::to_string(i), x, tpos, nullptr, hp_.voice_heads);
    const int64_t D = hp_.voice_dim, H = hp_.voice_heads, dh = D / H, nq = hp_.voice_tokens;
    ggml_tensor * q = W("voice.queries");                                                // [512, 32]
    for (int j = 0; j < 2; j++) {
        const std::string p = "voice.pool." + std::to_string(j);
        ggml_tensor * h = layer_norm(c, q, W(p + ".attn_norm.weight"), W(p + ".attn_norm.bias"), 1e-5f);
        ggml_tensor * qq = ggml_reshape_3d(c, linear(c, h, W(p + ".attn_q.weight"), W(p + ".attn_q.bias")), dh, H, nq);
        ggml_tensor * kv = linear(c, x, W(p + ".attn_kv.weight"), W(p + ".attn_kv.bias"));
        q = ggml_add(c, q, linear(c, attention(c, qq, part(c, kv, 0, dh, H), part(c, kv, 1, dh, H), nullptr),
                                  W(p + ".attn_out.weight"), W(p + ".attn_out.bias")));
        h = layer_norm(c, q, W(p + ".ffn_norm.weight"), W(p + ".ffn_norm.bias"), 1e-5f);
        h = linear(c, gelu(c, linear(c, h, W(p + ".ffn_up.weight"), W(p + ".ffn_up.bias"))),
                   W(p + ".ffn_down.weight"), W(p + ".ffn_down.bias"));
        q = ggml_add(c, q, h);
    }
    q = layer_norm(c, q, W("voice.out_norm.weight"), W("voice.out_norm.bias"), 1e-5f);
    ggml_tensor * tok = linear(c, q, W("voice.tokens_out.weight"), W("voice.tokens_out.bias"));   // [768, 32]
    ggml_tensor * m = ggml_reshape_2d(c, ggml_mean(c, ggml_cont(c, ggml_transpose(c, q))), D, 1);
    ggml_tensor * vecv = linear(c, ggml_silu(c, linear(c, m, W("voice.vec_fc1.weight"), W("voice.vec_fc1.bias"))),
                                W("voice.vec_fc2.weight"), W("voice.vec_fc2.bias"));
    g.out(tok);
    g.out(vecv);
    compute(g);
    tokens = get(tok);
    vec = get(vecv);
}

// The prompt (voice prefix) attends only to itself, so its keys/values per DiT block are computed once per voice.
// Also cached per voice: the voice tokens' cross-attention keys/values and the AdaNorm scale/shift of every block for
// each of the 4 sampler times (they depend only on the time and the voice vector).
bool Engine::build_prompt_cache(Voice & v) {
    const int P = std::min(hp_.prompt_frames, v.n_ref);
    const int D = hp_.dit_dim, H = hp_.dit_heads, dh = D / H, NT = hp_.voice_tokens, NL = hp_.dit_layers;
    const int R = 4 * NL + 2;                       // AdaNorm rows per sampler step
    if (v.buf) { ggml_backend_buffer_free(v.buf); v.buf = nullptr; }
    if (v.ctx) { ggml_free(v.ctx); v.ctx = nullptr; }
    ggml_init_params ip{ ggml_tensor_overhead() * (4 * NL + 8), nullptr, true };
    v.ctx = ggml_init(ip);
    v.kp.assign(NL, nullptr); v.vt.assign(NL, nullptr); v.xkp.assign(NL, nullptr); v.xvt.assign(NL, nullptr);
    for (int i = 0; i < NL; i++) {
        v.kp[i] = ggml_new_tensor_3d(v.ctx, GGML_TYPE_F32, dh, P, H);
        v.vt[i] = ggml_new_tensor_3d(v.ctx, GGML_TYPE_F32, P, dh, H);
        v.xkp[i] = ggml_new_tensor_3d(v.ctx, GGML_TYPE_F32, dh, NT, H);
        v.xvt[i] = ggml_new_tensor_3d(v.ctx, GGML_TYPE_F32, NT, dh, H);
    }
    v.ada = ggml_new_tensor_3d(v.ctx, GGML_TYPE_F32, D, R, 4);
    v.g_raw = ggml_new_tensor_1d(v.ctx, GGML_TYPE_F32, hp_.voice_out);
    v.buf = ggml_backend_alloc_ctx_tensors(v.ctx, main_backend());
    if (!v.buf) { err_ = "cannot allocate the voice cache"; return false; }
    v.n_prompt = P;

    Graph g(8192);
    ggml_context * c = g.ctx;
    std::vector<int32_t> pos(P);
    for (int i = 0; i < P; i++) pos[i] = i;
    ggml_tensor * tpos = g.in_i32(pos.data(), P);
    ggml_tensor * graw = g.in_f32(v.vec.data(), hp_.voice_out);
    ggml_tensor * tok = linear(c, g.in_f32(v.tokens.data(), hp_.voice_out, NT), W("dit.voice_tok_proj.weight"),
                               W("dit.voice_tok_proj.bias"));                                 // [384, 32]
    ggml_tensor * gd = linear(c, graw, W("dit.voice_vec_proj.weight"), W("dit.voice_vec_proj.bias"));   // [384]
    // AdaNorm parameters of the 4 sampler steps
    auto put = [&](ggml_tensor * src, int step, int row) {
        g.expand(ggml_cpy(c, src, ggml_view_1d(c, v.ada, D, ((size_t) step * R + row) * D * sizeof(float))));
    };
    for (int k = 0; k < 4; k++) {
        ggml_tensor * cond = time_cond(g, hp_.times[k], gd);
        ggml_tensor * s1, * sh;
        for (int i = 0; i < NL; i++) {
            const std::string p = "dit.blk." + std::to_string(i);
            ada_split(c, linear(c, cond, W(p + ".ada_attn.weight"), W(p + ".ada_attn.bias")), D, &s1, &sh);
            put(s1, k, 4 * i); put(sh, k, 4 * i + 1);
            ada_split(c, linear(c, cond, W(p + ".ada_ffn.weight"), W(p + ".ada_ffn.bias")), D, &s1, &sh);
            put(s1, k, 4 * i + 2); put(sh, k, 4 * i + 3);
        }
        ada_split(c, linear(c, cond, W("dit.out_ada.weight"), W("dit.out_ada.bias")), D, &s1, &sh);
        put(s1, k, 4 * NL); put(sh, k, 4 * NL + 1);
    }
    // the prompt itself, at t = 1
    ggml_tensor * cond1 = time_cond(g, 1.0f, gd);
    ggml_tensor * x = ggml_mul_mat(c, W("dit.in_prompt.weight"), g.in_f32(v.ref.data(), hp_.latent, P));
    x = ggml_add(c, ggml_add(c, x, W("dit.in_flag")), W("dit.in.bias"));
    for (int i = 0; i < NL; i++) {
        const std::string p = "dit.blk." + std::to_string(i);
        ggml_tensor * kv = linear(c, tok, W(p + ".xattn_kv.weight"), W(p + ".xattn_kv.bias"));
        ggml_tensor * xkp = ggml_cont(c, k_layout(c, part(c, kv, 0, dh, H)));
        ggml_tensor * xvt = v_layout(c, part(c, kv, 1, dh, H));
        g.expand(ggml_cpy(c, xkp, v.xkp[i]));
        g.expand(ggml_cpy(c, xvt, v.xvt[i]));
        DitMod m;
        ada_split(c, linear(c, cond1, W(p + ".ada_attn.weight"), W(p + ".ada_attn.bias")), D, &m.attn_scale1p, &m.attn_shift);
        ada_split(c, linear(c, cond1, W(p + ".ada_ffn.weight"), W(p + ".ada_ffn.bias")), D, &m.ffn_scale1p, &m.ffn_shift);
        ggml_tensor * ko = nullptr, * vo = nullptr;
        x = dit_block(g, i, x, m, tpos, nullptr, nullptr, xkp, xvt, &ko, &vo);
        g.expand(ggml_cpy(c, ko, v.kp[i]));
        g.expand(ggml_cpy(c, vo, v.vt[i]));
    }
    g.expand(ggml_cpy(c, graw, v.g_raw));
    compute(g);
    return true;
}

void Engine::read_prompt_cache(const Voice & v, int block, std::vector<float> & k, std::vector<float> & vv) const {
    // -> [P][H][dh] (the caller reorders to the reference layout)
    const int P = v.n_prompt, H = hp_.dit_heads, dh = hp_.dit_dim / H;
    const std::vector<float> kp = get(v.kp[block]), vt = get(v.vt[block]);   // [H][P][dh], [H][dh][P]
    k.assign((size_t) P * H * dh, 0.0f);
    vv.assign(k.size(), 0.0f);
    for (int h = 0; h < H; h++)
        for (int p = 0; p < P; p++)
            for (int d = 0; d < dh; d++) {
                k[((size_t) p * H + h) * dh + d] = kp[((size_t) h * P + p) * dh + d];
                vv[((size_t) p * H + h) * dh + d] = vt[((size_t) h * dh + d) * P + p];
            }
}

// ------------------------------------------------------------------------------------------------ text + durations

void Engine::text_and_durations(const std::vector<int32_t> & ids, const std::vector<int32_t> & lang,
                                const std::vector<float> & mask, const Voice & v,
                                std::vector<float> & h_out, std::vector<float> & dur_out) {
    const int N = (int) ids.size();
    Graph g(4096);
    ggml_context * c = g.ctx;
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; i++) pos[i] = i;
    std::vector<float> kmask((size_t) N * N);
    for (int q = 0; q < N; q++)
        for (int k = 0; k < N; k++) kmask[(size_t) q * N + k] = mask[k] > 0.5f ? 0.0f : -1e9f;
    ggml_tensor * tpos = g.in_i32(pos.data(), N);
    ggml_tensor * tmask = g.in_f32(kmask.data(), N, N);
    ggml_tensor * m = g.in_f32(mask.data(), 1, N);
    ggml_tensor * x = ggml_get_rows(c, W("text.tok_embd"), g.in_i32(ids.data(), N));
    x = ggml_add(c, x, ggml_get_rows(c, W("text.lang_embd"), g.in_i32(lang.data(), N)));
    for (int i = 0; i < hp_.text_layers; i++) x = tblock(g, "text.blk." + std::to_string(i), x, tpos, tmask, hp_.text_heads);
    ggml_tensor * h = ggml_mul(c, layer_norm(c, x, W("text.out_norm.weight"), W("text.out_norm.bias"), 1e-5f), m);
    // duration predictor
    ggml_tensor * gp = linear(c, v.g_raw, W("dur.voice_proj.weight"), W("dur.voice_proj.bias"));
    ggml_tensor * d = ggml_mul(c, ggml_add(c, h, gp), m);
    for (int i = 1; i <= 3; i++) {
        const std::string p = "dur.conv" + std::to_string(i);
        d = ggml_relu(c, linear(c, unfold(c, d, 3, 1), W(p + ".weight"), W(p + ".bias")));
        const std::string n = "dur.norm" + std::to_string(i);
        d = ggml_mul(c, layer_norm(c, d, W(n + ".weight"), W(n + ".bias"), 1e-5f), m);
    }
    d = linear(c, d, W("dur.out.weight"), W("dur.out.bias"));       // [1, N]
    g.out(h);
    g.out(d);
    compute(g);
    h_out = get(h);
    dur_out = get(d);
}

// ------------------------------------------------------------------------------------------------ DiT sampler

std::vector<float> Engine::sample(const Voice & v, const std::vector<float> & hf_new, int L, const NoiseFn & noise,
                                  SynthTrace * trace, const std::vector<float> * forced_in) {
    const int P = v.n_prompt, D = hp_.dit_dim, NL = hp_.dit_layers, R = 4 * NL + 2;
    const int lat = hp_.latent;
    Graph g(16384);
    ggml_context * c = g.ctx;
    std::vector<int32_t> pos(L);
    for (int i = 0; i < L; i++) pos[i] = P + i;
    ggml_tensor * tpos = g.in_i32(pos.data(), L);
    ggml_tensor * cond_text = linear(c, g.in_f32(hf_new.data(), hp_.text_dim, L), W("dit.text_proj.weight"),
                                     W("dit.text_proj.bias"));
    cond_text = ggml_add(c, cond_text, W("dit.in.bias"));
    std::vector<float> nz((size_t) L * lat);
    ggml_tensor * noise_t[4];
    for (int k = 0; k < 4; k++) {
        noise(k, L, lat, nz.data());
        noise_t[k] = g.in_f32(nz.data(), lat, L);
    }
    auto ada = [&](int step, int row) { return ggml_view_1d(c, v.ada, D, ((size_t) step * R + row) * D * sizeof(float)); };
    ggml_tensor * x = noise_t[0];
    ggml_tensor * outs_in[4], * outs_v[4], * outs_x1[4];
    for (int k = 0; k < 4; k++) {
        const float t = hp_.times[k];
        if (forced_in) x = g.in_f32(forced_in[k].data(), lat, L);
        ggml_tensor * hdn = ggml_add(c, ggml_mul_mat(c, W("dit.in_noisy.weight"), x), cond_text);
        for (int i = 0; i < NL; i++) {
            DitMod m{ ada(k, 4 * i), ada(k, 4 * i + 1), ada(k, 4 * i + 2), ada(k, 4 * i + 3) };
            hdn = dit_block(g, i, hdn, m, tpos, v.kp[i], v.vt[i], v.xkp[i], v.xvt[i], nullptr, nullptr);
        }
        ggml_tensor * vel = linear(c, modulate(c, hdn, ada(k, 4 * NL), ada(k, 4 * NL + 1)), W("dit.out.weight"),
                                   W("dit.out.bias"));                                    // [96, L]
        ggml_tensor * x1 = ggml_add(c, x, ggml_scale(c, vel, 1.0f - t));
        outs_in[k] = x; outs_v[k] = vel; outs_x1[k] = x1;
        if (trace) { g.out(x); g.out(vel); }
        g.out(x1);
        if (k + 1 < 4) {
            const float t2 = hp_.times[k + 1];
            x = ggml_add(c, ggml_scale(c, noise_t[k + 1], 1.0f - t2), ggml_scale(c, x1, t2));
        }
    }
    compute(g);
    if (trace) {
        for (int k = 0; k < 4; k++) {
            trace->step_in[k] = get(outs_in[k]);
            trace->step_v[k] = get(outs_v[k]);
            trace->step_x1[k] = get(outs_x1[k]);
        }
    }
    return get(outs_x1[3]);
}

// ------------------------------------------------------------------------------------------------ decoder

std::vector<float> Engine::decode(const std::vector<float> & z, int T) {
    Graph g(8192);
    ggml_context * c = g.ctx;
    ggml_tensor * x = g.in_f32(z.data(), hp_.latent, T);
    x = linear(c, unfold(c, x, 3, 1), W("dec.in.weight"), W("dec.in.bias"));          // [512, T]
    // ConvTranspose1d k8 s4 p2: two output phases, overlap-added one frame apart
    ggml_tensor * y0 = ggml_mul_mat(c, W("dec.up0.weight"), x);                        // [4*512, T]
    ggml_tensor * y1 = ggml_mul_mat(c, W("dec.up1.weight"), x);
    ggml_tensor * gsum = ggml_add(c, ggml_pad_ext(c, y0, 0, 0, 0, 1, 0, 0, 0, 0), ggml_pad_ext(c, y1, 0, 0, 1, 0, 0, 0, 0, 0));
    const int64_t C = hp_.dec_dim;
    gsum = ggml_reshape_2d(c, gsum, C, 4 * (T + 1));
    x = ggml_view_2d(c, gsum, C, 4 * (int64_t) T, gsum->nb[1], 2 * gsum->nb[1]);
    x = ggml_add(c, x, W("dec.up.bias"));                                              // [512, 4T]
    for (int i = 0; i < hp_.dec_blocks; i++) x = convnext(g, "dec.blk." + std::to_string(i), x);
    x = layer_norm(c, x, W("dec.out_norm.weight"), W("dec.out_norm.bias"), 1e-6f);
    x = linear(c, x, W("dec.head.weight"), W("dec.head.bias"));                        // [962, 4T]
    g.out(x);
    compute(g);
    const std::vector<float> head = get(x);
    return istft(head, 4 * T, T * hp_.samples_per_latent);
}

}  // namespace articulino
