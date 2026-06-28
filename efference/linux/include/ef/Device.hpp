////////////////////////////////////////////////////////////////////////////////
//
// File:      Device.hpp
// Purpose:   [PURPOSE]
// Author:    Calvin Nguyen, Gianluca Bencomo
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

#ifndef EF_DEVICE_HPP
#define EF_DEVICE_HPP

#include "Enums.hpp"
#include "Parameters.hpp"
#include <functional>
#include <memory>
#include <string>

namespace ef {

using StreamEndCallback = std::function<void(const StreamResult&)>;
using StreamSink = std::function<void(const uint8_t* data, size_t len)>;

// --- device handle -------------------------------------------

struct InitParameters {
    int verbose   = 0;  // >0 → log transport detail to stderr
    int device_id = 0;  // index among attached M1s (0 = first)
};

class Device {
public:
    Device();
    ~Device();
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    ERROR_CODE open(const InitParameters& params = InitParameters{});

    InitParameters get_init_parameters();


    DeviceInformation get_device_information();

    ConfigureResult configure(const Configuration& cfg);

    StreamResult stream_mcap(const std::string& path,
                             const std::function<bool()>& should_stop,
                             const StreamEndCallback& on_end = {});

    StreamResult stream_mcap(const std::string& path, double seconds,
                             const StreamEndCallback& on_end = {});

    StreamResult stream_mcap(const StreamSink& sink,
                             const std::function<bool()>& should_stop,
                             const StreamEndCallback& on_end = {});

    // Release the interface + handle. Safe to call when not open.
    void close();

    bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* to_string(ERROR_CODE);
const char* to_string(STREAM_END);
const char* to_string(MODEL);
const char* to_string(INPUT_TYPE);
const char* to_string(RESOLUTION);

}  // namespace ef

#endif  // EF_DEVICE_HPP