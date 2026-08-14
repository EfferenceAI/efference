////////////////////////////////////////////////////////////////////////////////
//
// File:      internal/recording_paths.hpp
// Purpose:   Destination and URL validation for recording transfers.
//
// Internal. Header-only and free of Device state.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_INTERNAL_RECORDING_PATHS_HPP
#define EF_INTERNAL_RECORDING_PATHS_HPP

#include <sys/stat.h>

#include <cctype>
#include <cstring>
#include <string>

#include <ef/Enums.hpp>

namespace ef {
namespace internal {

// A recording name is a session name, never a path, so it must be usable as one
// filename component and nothing more.
inline bool name_is_plain(const std::string& name) {
    return !name.empty()
        && name.find('/')  == std::string::npos
        && name.find('\\') == std::string::npos
        && name != ".." && name != "."
        && name.find("../") == std::string::npos;
}

// Resolve where a download lands: an existing directory receives "<name>.mcap",
// any other path is used as given. Returns INVALID_FUNCTION_CALL on an empty
// argument or a name that is not a plain session name.
inline ERROR_CODE resolve_download_dest(const std::string& name,
                                        const std::string& dest_path,
                                        std::string& out) {
    if (dest_path.empty() || !name_is_plain(name)) return ERROR_CODE::INVALID_FUNCTION_CALL;

    out = dest_path;
    struct stat ds;
    if (::stat(out.c_str(), &ds) == 0 && S_ISDIR(ds.st_mode)) {
        if (out.back() != '/') out += '/';
        out += name + ".mcap";
    }
    return ERROR_CODE::SUCCESS;
}

// Only http(s) is accepted; the scheme match is case-insensitive.
inline bool is_http_url(const std::string& u) {
    auto starts_with_ci = [&](const char* p) {
        std::size_t n = std::strlen(p);
        if (u.size() < n) return false;
        for (std::size_t i = 0; i < n; ++i)
            if (std::tolower((unsigned char)u[i]) != p[i]) return false;
        return true;
    };
    return starts_with_ci("http://") || starts_with_ci("https://");
}

}  // namespace internal
}  // namespace ef

#endif  // EF_INTERNAL_RECORDING_PATHS_HPP
