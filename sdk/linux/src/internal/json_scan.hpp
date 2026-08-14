////////////////////////////////////////////////////////////////////////////////
//
// File:      internal/json_scan.hpp
// Purpose:   Targeted scan of the flat update-check response.
//
// Internal. A flat object with known keys, so a targeted scan avoids adding a JSON
// library to a public SDK. An absent key leaves the caller's default: every
// response field is optional.
//
// Header-only and free of transport state so it can be exercised directly; reached
// through check_for_update() it sits behind a live HTTP call.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_INTERNAL_JSON_SCAN_HPP
#define EF_INTERNAL_JSON_SCAN_HPP

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ef {
namespace internal {

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {   // other control chars -> \u00XX
                    char buf[7];
                    std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ---- response scanning ---------------------------------------------------------------
// A flat object with known keys, so a targeted scan avoids adding a JSON library to a
// public SDK. An absent key leaves the caller's default: every response field is optional.

// Offset just past `"key"` and its colon, or npos.
inline size_t find_value(const std::string& j, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t k = j.find(needle);
    if (k == std::string::npos) return std::string::npos;
    k = j.find(':', k + needle.size());
    if (k == std::string::npos) return std::string::npos;
    return j.find_first_not_of(" \t\r\n", k + 1);
}

// A JSON string value, unescaped. False when absent or not a string, so a null
// download_url reads as "no URL" rather than the literal text "null".
inline bool get_string(const std::string& j, const char* key, std::string* out) {
    size_t p = find_value(j, key);
    if (p == std::string::npos || j[p] != '"') return false;
    std::string v;
    for (++p; p < j.size() && j[p] != '"'; ++p) {
        if (j[p] != '\\') { v += j[p]; continue; }
        if (++p >= j.size()) return false;
        switch (j[p]) {
            case 'n': v += '\n'; break;
            case 'r': v += '\r'; break;
            case 't': v += '\t'; break;
            case 'u': {                       // \uXXXX: keep ASCII, drop the rest
                if (p + 4 >= j.size()) return false;
                const long cp = std::strtol(j.substr(p + 1, 4).c_str(), nullptr, 16);
                if (cp >= 0x20 && cp < 0x7f) v += (char)cp;
                p += 4;
                break;
            }
            default: v += j[p];               // covers \" \\ \/ and anything odd
        }
    }
    if (p >= j.size()) return false;          // unterminated string
    *out = v;
    return true;
}

inline bool get_uint(const std::string& j, const char* key, unsigned int* out) {
    size_t p = find_value(j, key);
    if (p == std::string::npos) return false;
    if (j[p] < '0' || j[p] > '9') return false;      // negatives/null are not versions
    *out = (unsigned int)std::strtoul(j.c_str() + p, nullptr, 10);
    return true;
}

inline bool get_bool(const std::string& j, const char* key, bool* out) {
    size_t p = find_value(j, key);
    if (p == std::string::npos) return false;
    if (!j.compare(p, 4, "true"))  { *out = true;  return true; }
    if (!j.compare(p, 5, "false")) { *out = false; return true; }
    return false;
}

}  // namespace internal
}  // namespace ef

#endif  // EF_INTERNAL_JSON_SCAN_HPP
