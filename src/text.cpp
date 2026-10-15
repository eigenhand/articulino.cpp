// Text: the model's letter normalizer (as text.normalisiere), [en] markup (as sprich.zerlege) and the sentence
// splitter of the reference service (server.saetze). The full German frontend (numbers, dates, units, ...) is not
// part of the engine; it runs before it (see README).
#include "engine.h"

#include <algorithm>
#include <cstring>

namespace articulino {

namespace {

struct CharMap { uint32_t cp; const char * out; };
struct CodeRange { uint32_t lo, hi; };
#include "text_table.inc"

const char * k_charset = " abcdefghijklmnopqrstuvwxyz\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f.,?!-'\":;()0123456789";

std::u32string utf8_to_u32(const std::string & s) {
    std::u32string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t) s[i];
        uint32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { i++; out.push_back(0xFFFD); continue; }
        if (i + n > s.size()) { out.push_back(0xFFFD); break; }
        bool ok = true;
        for (int k = 1; k < n; k++) {
            const uint8_t cc = (uint8_t) s[i + k];
            if ((cc >> 6) != 2) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); i++; continue; }
        out.push_back(cp);
        i += n;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string & s) {
    std::string out;
    for (char32_t cp : s) {
        if (cp < 0x80) out += (char) cp;
        else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F));
        }
    }
    return out;
}

bool in_ranges(const CodeRange * r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_space(char32_t cp) { return in_ranges(k_space_ranges, sizeof(k_space_ranges) / sizeof(k_space_ranges[0]), cp); }
bool is_alpha(char32_t cp) { return in_ranges(k_alpha_ranges, sizeof(k_alpha_ranges) / sizeof(k_alpha_ranges[0]), cp); }

const std::u32string & charset32() {
    static const std::u32string cs = utf8_to_u32(k_charset);
    return cs;
}

int symbol_id(char32_t cp) {
    const std::u32string & cs = charset32();
    for (size_t i = 0; i < cs.size(); i++)
        if (cs[i] == cp) return (int) i + 1;
    return 0;
}

// output of normalisiere() for a single code point
std::u32string map_char(char32_t cp) {
    const size_t n = sizeof(k_char_map) / sizeof(k_char_map[0]);
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (k_char_map[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n && k_char_map[lo].cp == cp) return utf8_to_u32(k_char_map[lo].out);
    if (symbol_id(cp)) return std::u32string(1, cp);
    return std::u32string();
}

std::u32string normalize32(const std::u32string & in) {
    std::u32string raw;
    for (char32_t cp : in) raw += map_char(cp);
    // " ".join(raw.split()): collapse runs of spaces, strip both ends
    std::u32string out;
    bool pending = false;
    for (char32_t cp : raw) {
        if (cp == U' ') { pending = !out.empty(); continue; }
        if (pending) { out += U' '; pending = false; }
        out += cp;
    }
    return out;
}

std::u32string strip32(const std::u32string & s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) b++;
    while (e > b && is_space(s[e - 1])) e--;
    return s.substr(b, e - b);
}

size_t count_alpha(const std::u32string & s) {
    size_t n = 0;
    for (char32_t c : s) n += is_alpha(c) ? 1 : 0;
    return n;
}

}  // namespace

std::string normalize_text(const std::string & utf8) {
    return u32_to_utf8(normalize32(utf8_to_u32(utf8)));
}

bool is_alnum_symbol(int32_t id) {
    if (id <= 0) return false;
    const std::u32string & cs = charset32();
    if ((size_t) id > cs.size()) return false;
    const char32_t c = cs[id - 1];
    return (c >= U'a' && c <= U'z') || (c >= U'0' && c <= U'9') || c == 0xE4 || c == 0xF6 || c == 0xFC || c == 0xDF;
}

// as sprich.zerlege(): [en]..[/en] and [de]..[/de] switch the language of the letters in between
void text_to_ids(const std::string & utf8, int base_lang, std::vector<int32_t> & ids, std::vector<int32_t> & lang) {
    ids.clear();
    lang.clear();
    const std::u32string s = utf8_to_u32(utf8);
    const int space = symbol_id(U' ');
    int cur = base_lang;
    std::vector<std::pair<std::u32string, int>> parts;   // (text, language) between tags
    size_t i = 0, start = 0;
    auto tag_at = [&](size_t p, int & tag_lang, bool & closing, size_t & len) {
        static const char32_t * tags[] = { U"[en]", U"[de]", U"[/en]", U"[/de]" };
        for (int t = 0; t < 4; t++) {
            const std::u32string tg(tags[t]);
            if (s.compare(p, tg.size(), tg) == 0) {
                tag_lang = (t % 2 == 0) ? 1 : 0;
                closing = t >= 2;
                len = tg.size();
                return true;
            }
        }
        return false;
    };
    while (i <= s.size()) {
        int tl = 0; bool closing = false; size_t len = 0;
        const bool at_end = i == s.size();
        if (at_end || (s[i] == U'[' && tag_at(i, tl, closing, len))) {
            parts.emplace_back(s.substr(start, i - start), cur);
            if (at_end) break;
            cur = closing ? base_lang : tl;
            i += len;
            start = i;
        } else {
            i++;
        }
    }
    for (auto & part : parts) {
        const std::u32string n = normalize32(part.first);
        if (n.empty()) continue;
        if (!ids.empty() && !part.first.empty() && is_space(part.first.front()) && ids.back() != space) {
            ids.push_back(space); lang.push_back(part.second);
        }
        for (char32_t c : n) { ids.push_back(symbol_id(c)); lang.push_back(part.second); }
        if (!part.first.empty() && is_space(part.first.back())) { ids.push_back(space); lang.push_back(part.second); }
    }
    while (!ids.empty() && ids.back() == space) { ids.pop_back(); lang.pop_back(); }
}

int markup_language_after(const std::string & s, int state) {
    size_t p = 0;
    while ((p = s.find('[', p)) != std::string::npos) {
        if (s.compare(p, 4, "[en]") == 0) state = 1;
        else if (s.compare(p, 4, "[de]") == 0) state = 0;
        else if (s.compare(p, 5, "[/en]") == 0 || s.compare(p, 5, "[/de]") == 0) state = -1;
        p++;
    }
    return state;
}

// server.saetze(): finished sentences from the buffer; short ones (< 20 letters) wait for the next one,
// long ones (> 180 characters) are split at a comma (or a space) between characters 60 and 180
std::vector<std::string> split_sentences(const std::string & buffer, bool all, std::string & rest_out) {
    const std::u32string b = utf8_to_u32(buffer);
    auto is_term = [](char32_t c) { return c == U'.' || c == U'!' || c == U'?' || c == 0x2026 || c == U':' || c == U';'; };
    auto is_close = [](char32_t c) { return c == U'"' || c == U'\'' || c == U')' || c == U']'; };
    // one match of (.+?[.!?…:;]+["')\]]*)(?:\s+|$) (re.S) starting at pos: group end and match end
    auto next_match = [&](size_t pos, size_t & grp_end, size_t & match_end) {
        for (size_t e = pos + 1; e < b.size(); e++) {
            if (!is_term(b[e])) continue;
            size_t p = e;
            while (p < b.size() && is_term(b[p])) p++;
            while (p < b.size() && is_close(b[p])) p++;
            if (p == b.size()) { grp_end = match_end = p; return true; }
            if (!is_space(b[p])) continue;
            grp_end = p;
            while (p < b.size() && is_space(b[p])) p++;
            match_end = p;
            return true;
        }
        return false;
    };
    std::vector<std::u32string> found;
    size_t pos = 0, ge = 0, me = 0;
    while (pos < b.size() && next_match(pos, ge, me)) {
        if (me == b.size() && !all && !is_space(b.back())) break;   // punctuation at the end: more may follow
        found.push_back(strip32(b.substr(pos, ge - pos)));
        pos = me;
    }
    const size_t rest_pos = pos;
    std::u32string rest = b.substr(std::min(rest_pos, b.size()));
    if (all && !strip32(rest).empty()) { found.push_back(strip32(rest)); rest.clear(); }
    std::vector<std::u32string> merged;
    for (auto & s : found) {
        if (!merged.empty() && count_alpha(merged.back()) < 20) merged.back() += U" " + s;
        else merged.push_back(s);
    }
    if (!merged.empty() && !all && count_alpha(merged.back()) < 20) {
        rest = merged.back() + U" " + rest;
        merged.pop_back();
    }
    const size_t LANG = 180;
    std::vector<std::string> out;
    for (auto s : merged) {
        while (s.size() > LANG) {
            // s.rfind(", ", 60, LANG): last ", " that lies completely inside s[60:180]
            long k = -1;
            for (long j = (long) LANG - 2; j >= 60; j--)
                if (s[j] == U',' && s[j + 1] == U' ') { k = j; break; }
            if (k > 0) k = k + 1;
            else {
                k = -1;
                for (long j = (long) LANG - 1; j >= 60; j--)
                    if (s[j] == U' ') { k = j; break; }
            }
            if (k <= 0) break;
            out.push_back(u32_to_utf8(strip32(s.substr(0, k))));
            s = strip32(s.substr(k));
        }
        out.push_back(u32_to_utf8(s));
    }
    rest_out = u32_to_utf8(rest);
    return out;
}

}  // namespace articulino
