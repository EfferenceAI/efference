////////////////////////////////////////////////////////////////////////////////
//
// File:      update_check.cpp
// Purpose:   Host-side update-check client: ask the service what a device should
//            be running, and hand the answer back as an UpdateAvailability.
// Author:    Calvin Nguyen
//
// Copyright (c) 2026, Remnant Robotics, Inc. All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////////////////////////////////////////////////////////////

#include <curl/curl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "ef/Device.hpp"

namespace ef {

namespace {

// Compiled-in fallback endpoint, overridable at build time.
#ifndef EF_UPDATE_CHECK_URL_DEFAULT
#define EF_UPDATE_CHECK_URL_DEFAULT \
    "https://axvfph7h1l.execute-api.us-west-1.amazonaws.com/ota/check"
#endif

constexpr long kTimeoutSeconds = 15;
constexpr size_t kMaxBodyBytes = 64 * 1024;   // a real answer is a few hundred bytes

// curl_easy_init() will call curl_global_init() implicitly, but that is not thread-safe.
// check_for_update is public API a fleet tool may well call from several threads at once,
// so serialise the one-time global init here. No matching curl_global_cleanup: a library
// cannot know when the process is finished with libcurl.
std::once_flag g_curl_once;
void curl_init_once() {
    std::call_once(g_curl_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

size_t collect(char* ptr, size_t size, size_t nmemb, void* userdata) {
    std::string* body = static_cast<std::string*>(userdata);
    const size_t n = size * nmemb;
    if (body->size() + n > kMaxBodyBytes) return 0;   // short return aborts the transfer
    body->append(ptr, n);
    return n;
}

std::string json_escape(const std::string& s) {
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
size_t find_value(const std::string& j, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t k = j.find(needle);
    if (k == std::string::npos) return std::string::npos;
    k = j.find(':', k + needle.size());
    if (k == std::string::npos) return std::string::npos;
    return j.find_first_not_of(" \t\r\n", k + 1);
}

// A JSON string value, unescaped. False when absent or not a string, so a null
// download_url reads as "no URL" rather than the literal text "null".
bool get_string(const std::string& j, const char* key, std::string* out) {
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

bool get_uint(const std::string& j, const char* key, unsigned int* out) {
    size_t p = find_value(j, key);
    if (p == std::string::npos) return false;
    if (j[p] < '0' || j[p] > '9') return false;      // negatives/null are not versions
    *out = (unsigned int)std::strtoul(j.c_str() + p, nullptr, 10);
    return true;
}

bool get_bool(const std::string& j, const char* key, bool* out) {
    size_t p = find_value(j, key);
    if (p == std::string::npos) return false;
    if (!j.compare(p, 4, "true"))  { *out = true;  return true; }
    if (!j.compare(p, 5, "false")) { *out = false; return true; }
    return false;
}

}  // namespace

std::string update_check_url() {
    const char* env = std::getenv("EF_UPDATE_CHECK_URL");
    if (env && *env) return env;
    return EF_UPDATE_CHECK_URL_DEFAULT;
}

ERROR_CODE check_for_update(const DeviceInformation& info, UpdateAvailability& out,
                            const std::string& channel,
                            const std::string& service_url) {
    out = UpdateAvailability{};

    // Never guess the model: inventing "M1" is how a future board gets offered the
    // wrong firmware, and it defeats the service's unsupported_model guard.
    if (info.model_name.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;

    const std::string url = service_url.empty() ? update_check_url() : service_url;

    char body[1024];
    const int blen = std::snprintf(body, sizeof body,
                  "{\"model\":\"%s\",\"current_version\":%u,\"hw_rev\":\"%s\","
                  "\"serial\":\"%s\",\"channel\":\"%s\"}",
                  json_escape(info.model_name).c_str(), info.firmware_version,
                  json_escape(info.hw_rev).c_str(), json_escape(info.serial).c_str(),
                  json_escape(channel).c_str());
    // snprintf truncates silently, which would send malformed JSON. No real identity
    // comes close to the buffer, so this means a caller passed something absurd.
    if (blen < 0 || (size_t)blen >= sizeof body) {
        out.service_error = "request too large to encode";
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    }

    curl_init_once();
    CURL* c = curl_easy_init();
    if (!c) return ERROR_CODE::COMMUNICATION_ERROR;

    std::string resp;
    curl_slist* hdrs = curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, collect);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, kTimeoutSeconds);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);   // safe to call from a threaded host
    const CURLcode rc = curl_easy_perform(c);
    long http = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(c);

    if (rc != CURLE_OK) {
        out.service_error = curl_easy_strerror(rc);
        return ERROR_CODE::COMMUNICATION_ERROR;
    }

    // Carry up the service's {error, details}: the ERROR_CODE cannot tell a bad request
    // from a broken backend.
    std::string err, detail;
    if (get_string(resp, "error", &err)) {
        get_string(resp, "details", &detail);
        out.service_error = detail.empty() ? err : err + ": " + detail;
    }

    // 404 = nothing published for this device: no update, not a failure. Not matched on
    // the error string, which is not part of the contract.
    if (http == 404) return ERROR_CODE::SUCCESS;
    if (http != 200) {
        if (out.service_error.empty())
            out.service_error = "HTTP " + std::to_string(http);
        return ERROR_CODE::COMMUNICATION_ERROR;
    }

    if (!get_bool(resp, "update_available", &out.available)) {
        out.service_error = "unrecognized response (no update_available field)";
        return ERROR_CODE::COMMUNICATION_ERROR;      // not a response we understand
    }
    get_uint(resp, "target_version", &out.target_version);
    get_string(resp, "target_version_str", &out.target_version_str);
    get_string(resp, "notes", &out.notes);
    get_string(resp, "download_url", &out.url);

    // "yes" with a missing or null URL is not an available update.
    if (out.available && out.url.empty()) out.available = false;
    return ERROR_CODE::SUCCESS;
}

}  // namespace ef
