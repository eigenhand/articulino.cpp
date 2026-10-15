// Prints the normalizer output and the sentence split for lines of input, for comparison with the Python reference
// (tests/check_text.py). Each input line: "N|text" (normalize) or "S|text" (split, all=1) or "P|text" (split, all=0).
#include "engine.h"

#include <cstdio>
#include <iostream>
#include <string>

using namespace articulino;

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() < 2) continue;
        const char mode = line[0];
        std::string text = line.substr(2);
        // \n in the input stands for a newline
        for (size_t p; (p = text.find("\\n")) != std::string::npos;) text.replace(p, 2, "\n");
        if (mode == 'N') {
            std::printf("%s\n", normalize_text(text).c_str());
        } else {
            std::string rest;
            auto s = split_sentences(text, mode == 'S', rest);
            std::string out;
            for (auto & x : s) out += "[" + x + "]";
            std::printf("%s|rest=[%s]\n", out.c_str(), rest.c_str());
        }
    }
    return 0;
}
