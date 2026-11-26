#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::frontend {

// Synthesizes canonical Qwen3.8 XML tool-call skeleton spans from declared tools so the per-request
// SuffixAutomaton can draft function/parameter boilerplate from Turn 1 onward without altering
// prompt tokens or target probabilities.
inline std::vector<std::string>
ngram_tool_schema_sources(std::span<const ToolCallOutputContract::Tool> tools) {
    std::vector<std::string> sources;
    if (tools.empty()) { return sources; }
    sources.reserve(tools.size() * 2 + 1);
    for (const auto& tool : tools) {
        if (tool.name.empty()) { continue; }
        if (tool.parameters.empty()) {
            sources.push_back("<tool_call>\n<function=" + tool.name + ">\n</function>\n</tool_call>");
            continue;
        }
        sources.push_back("<tool_call>\n<function=" + tool.name + ">\n<parameter=" + tool.parameters[0].name + ">\n");
        for (std::size_t i = 1; i < tool.parameters.size(); ++i) {
            sources.push_back("\n</parameter>\n<parameter=" + tool.parameters[i].name + ">\n");
        }
    }
    if (!sources.empty()) { sources.push_back("\n</parameter>\n</function>\n</tool_call>"); }
    return sources;
}

// Called only for tool-result text. Recognize consecutive numbered runs, never
// rewrite target messages. Preserve code indentation after exactly one separator.
inline std::vector<std::string> ngram_numbered_sources(std::string_view text) {
    struct Line {
        std::uint64_t number;
        std::string_view body;
        std::string_view style;
    };

    const auto parse = [](std::string_view line) -> std::optional<Line> {
        while (!line.empty() && line.front() == ' ') { line.remove_prefix(1); }
        if (!line.empty() && line.front() == 'L') { line.remove_prefix(1); }
        std::uint64_t number = 0;
        const auto parsed    = std::from_chars(line.data(), line.data() + line.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr == line.data()) { return std::nullopt; }
        auto rest = line.substr(static_cast<std::size_t>(parsed.ptr - line.data()));
        for (std::string_view delimiter : {": ", "\t", "\xe2\x86\x92", " | ", "| "}) {
            if (rest.starts_with(delimiter)) {
                return Line{number, rest.substr(delimiter.size()), delimiter};
            }
        }
        return std::nullopt;
    };
    std::vector<std::string> sources;
    std::string run;
    std::uint64_t previous = 0;
    std::string_view style;
    std::size_t count = 0;
    const auto flush  = [&] {
        if (count >= 3) { sources.push_back(std::move(run)); }
        run.clear();
        count = 0;
    };
    while (!text.empty()) {
        auto length        = text.find('\n');
        const bool newline = length != std::string_view::npos;
        if (!newline) { length = text.size(); }
        const auto parsed = parse(text.substr(0, length));
        if (!parsed) {
            flush();
        } else {
            if (count != 0 && (previous == std::numeric_limits<std::uint64_t>::max() ||
                               parsed->number != previous + 1 || parsed->style != style)) {
                flush();
            }
            run.append(parsed->body);
            if (newline) { run.push_back('\n'); }
            previous = parsed->number;
            style    = parsed->style;
            ++count;
        }
        text.remove_prefix(length + (newline ? 1 : 0));
    }
    flush();
    return sources;
}

// Extracts recurring structural entities from prompt message text for the L3-resident schema/entity
// proposer: file paths, quoted JSON key/value fragments, and snake_case/camelCase identifiers.
// Deduplicated and bounded (at most 64 snippets, 4096 total characters) so structural ingestion
// stays compact and strictly inside L3.
inline std::vector<std::string> ngram_prompt_entity_sources(std::span<const std::string_view> texts) {
    constexpr std::size_t kMaxSnippets = 64U;
    constexpr std::size_t kMaxChars    = 4096U;
    std::vector<std::string> sources;
    std::unordered_set<std::string> seen;
    std::size_t total_chars = 0;

    const auto try_add = [&](std::string_view sv) -> bool {
        if (sv.empty() || sv.size() > 160U) { return true; }
        if (sources.size() >= kMaxSnippets || total_chars + sv.size() > kMaxChars) { return false; }
        std::string key(sv);
        if (seen.insert(key).second) {
            total_chars += key.size();
            sources.push_back(std::move(key));
        }
        return sources.size() < kMaxSnippets && total_chars < kMaxChars;
    };

    const auto is_ident_char = [](unsigned char ch) noexcept {
        return std::isalnum(ch) != 0 || ch == '_';
    };
    const auto is_path_char = [](unsigned char ch) noexcept {
        return std::isalnum(ch) != 0 || ch == '_' || ch == '.' || ch == '/' || ch == '-';
    };

    for (std::string_view text : texts) {
        if (sources.size() >= kMaxSnippets || total_chars >= kMaxChars) { break; }
        for (std::size_t i = 0; i < text.size();) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            // 1. Quoted JSON key/value spans ("key": "value" or "key": number/bool or "value")
            if (c == '"') {
                const std::size_t close = text.find('"', i + 1);
                if (close != std::string_view::npos && close > i + 1 && close - i <= 96U) {
                    std::size_t span_end = close + 1;
                    // Extend through `: "value"` or `: primitive` when present right after closing quote.
                    std::size_t j = span_end;
                    while (j < text.size() && (text[j] == ' ' || text[j] == '\t')) { ++j; }
                    if (j < text.size() && text[j] == ':') {
                        ++j;
                        while (j < text.size() && (text[j] == ' ' || text[j] == '\t')) { ++j; }
                        if (j < text.size() && text[j] == '"') {
                            const std::size_t vclose = text.find('"', j + 1);
                            if (vclose != std::string_view::npos && vclose - i <= 144U) {
                                span_end = vclose + 1;
                            }
                        } else if (j < text.size()) {
                            std::size_t k = j;
                            while (k < text.size() && (std::isalnum(static_cast<unsigned char>(text[k])) != 0 ||
                                                       text[k] == '_' || text[k] == '-' || text[k] == '.')) {
                                ++k;
                            }
                            if (k > j && k - i <= 128U) { span_end = k; }
                        }
                    }
                    const std::string_view quoted = text.substr(i, span_end - i);
                    if (quoted.size() >= 4U && !try_add(quoted)) { return sources; }
                    i = span_end;
                    continue;
                }
            }
            // 2. File paths or multi-component identifiers
            if (is_path_char(c)) {
                std::size_t end = i + 1;
                while (end < text.size() && is_path_char(static_cast<unsigned char>(text[end]))) { ++end; }
                while (end > i && (text[end - 1] == '.' || text[end - 1] == '/')) { --end; }
                const std::string_view tok = text.substr(i, end - i);
                if (tok.size() >= 4U) {
                    const std::size_t dot = tok.rfind('.');
                    const bool has_slash  = tok.find('/') != std::string_view::npos;
                    const bool has_ext    = dot != std::string_view::npos && dot > 0 && dot + 1 < tok.size() &&
                                            tok.size() - dot - 1 <= 8U && std::isalpha(static_cast<unsigned char>(tok[dot + 1])) != 0;
                    if (has_slash && has_ext) {
                        if (!try_add(tok)) { return sources; }
                        i = end;
                        continue;
                    }
                }
            }
            // 3. snake_case or camelCase identifiers >= 6 chars
            if (std::isalpha(c) != 0 || c == '_') {
                std::size_t end = i + 1;
                bool has_under  = (c == '_');
                bool has_camel  = false;
                bool prev_lower = std::islower(c) != 0;
                while (end < text.size() && is_ident_char(static_cast<unsigned char>(text[end]))) {
                    const unsigned char ch = static_cast<unsigned char>(text[end]);
                    has_under              = has_under || ch == '_';
                    has_camel              = has_camel || (prev_lower && std::isupper(ch) != 0);
                    prev_lower             = std::islower(ch) != 0;
                    ++end;
                }
                const std::size_t len = end - i;
                if (len >= 6U && len <= 80U && (has_under || has_camel)) {
                    if (!try_add(text.substr(i, len))) { return sources; }
                }
                i = end;
                continue;
            }
            ++i;
        }
    }
    return sources;
}

} // namespace infernix::models::qwen3_5::frontend
