// json.hpp — small self-contained JSON value + recursive-descent parser.
//
// Internal to the SDK (no external dependency). Enough to walk the device's
// get_device_information reply (nested objects, string/number/bool arrays).
// Order-preserving objects; UTF-8 output for \uXXXX (BMP). Throws
// std::runtime_error on malformed input.
#ifndef EF_JSON_HPP
#define EF_JSON_HPP

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ef {
namespace detail {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type        type = Type::Null;
    bool        b    = false;
    double      num  = 0.0;
    std::string str;
    std::vector<Json>                          arr;
    std::vector<std::pair<std::string, Json>>  obj;

    bool is_null()   const { return type == Type::Null; }
    bool is_bool()   const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array()  const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool contains(const std::string& k) const {
        if (type != Type::Object) return false;
        for (const auto& kv : obj)
            if (kv.first == k) return true;
        return false;
    }

    // Object lookup; returns a static null Json for a missing key (so chains
    // like j["a"]["b"].as_int() never throw — they degrade to defaults).
    const Json& operator[](const std::string& k) const {
        static const Json kNull;
        if (type == Type::Object)
            for (const auto& kv : obj)
                if (kv.first == k) return kv.second;
        return kNull;
    }

    int         as_int(int def = 0)                         const { return is_number() ? (int)num : def; }
    double      as_double(double def = 0.0)                 const { return is_number() ? num : def; }
    bool        as_bool(bool def = false)                   const { return is_bool() ? b : def; }
    std::string as_string(const std::string& def = "")      const { return is_string() ? str : def; }

    static Json parse(const std::string& s);
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    Json parse() {
        ws();
        Json v = value();
        ws();
        return v;
    }

private:
    const std::string& s_;
    size_t             i_ = 0;

    [[noreturn]] void err(const std::string& m) {
        throw std::runtime_error("json: " + m + " at offset " + std::to_string(i_));
    }

    void ws() {
        while (i_ < s_.size() &&
               (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
            i_++;
    }

    void expect(const char* lit) {
        for (const char* p = lit; *p; ++p) {
            if (i_ >= s_.size() || s_[i_] != *p) err("expected literal");
            i_++;
        }
    }

    Json value() {
        ws();
        if (i_ >= s_.size()) err("unexpected end of input");
        char c = s_[i_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { Json j; j.type = Json::Type::String; j.str = string(); return j; }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') { expect("null"); return Json{}; }
        return number();
    }

    Json boolean() {
        Json j;
        j.type = Json::Type::Bool;
        if (s_[i_] == 't') { expect("true");  j.b = true;  }
        else               { expect("false"); j.b = false; }
        return j;
    }

    Json number() {
        size_t st = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) i_++;
        while (i_ < s_.size() &&
               (std::isdigit((unsigned char)s_[i_]) || s_[i_] == '.' ||
                s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-'))
            i_++;
        if (i_ == st) err("invalid number");
        Json j;
        j.type = Json::Type::Number;
        j.num  = std::strtod(s_.substr(st, i_ - st).c_str(), nullptr);
        return j;
    }

    std::string string() {
        if (s_[i_] != '"') err("expected string");
        i_++;
        std::string out;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i_ >= s_.size()) err("dangling escape");
                char e = s_[i_++];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) err("short \\u escape");
                        unsigned v = (unsigned)std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16);
                        i_ += 4;
                        if (v < 0x80) {
                            out += (char)v;
                        } else if (v < 0x800) {
                            out += (char)(0xC0 | (v >> 6));
                            out += (char)(0x80 | (v & 0x3F));
                        } else {
                            out += (char)(0xE0 | (v >> 12));
                            out += (char)(0x80 | ((v >> 6) & 0x3F));
                            out += (char)(0x80 | (v & 0x3F));
                        }
                        break;
                    }
                    default: err("invalid escape");
                }
            } else {
                out += c;
            }
        }
        err("unterminated string");
    }

    Json object() {
        Json j;
        j.type = Json::Type::Object;
        i_++;  // '{'
        ws();
        if (i_ < s_.size() && s_[i_] == '}') { i_++; return j; }
        for (;;) {
            ws();
            std::string k = string();
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') err("expected ':'");
            i_++;
            Json v = value();
            j.obj.emplace_back(std::move(k), std::move(v));
            ws();
            if (i_ >= s_.size()) err("unterminated object");
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == '}') { i_++; break; }
            err("expected ',' or '}'");
        }
        return j;
    }

    Json array() {
        Json j;
        j.type = Json::Type::Array;
        i_++;  // '['
        ws();
        if (i_ < s_.size() && s_[i_] == ']') { i_++; return j; }
        for (;;) {
            Json v = value();
            j.arr.push_back(std::move(v));
            ws();
            if (i_ >= s_.size()) err("unterminated array");
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == ']') { i_++; break; }
            err("expected ',' or ']'");
        }
        return j;
    }
};

inline Json Json::parse(const std::string& s) {
    JsonParser p(s);
    return p.parse();
}

}  // namespace detail
}  // namespace ef

#endif  // EF_JSON_HPP
