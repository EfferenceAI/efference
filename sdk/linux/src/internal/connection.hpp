////////////////////////////////////////////////////////////////////////////////
//
// File:      connection.hpp
// Purpose:   Abstract control/stream connection interface (internal).
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

#ifndef EF_CONNECTION_HPP
#define EF_CONNECTION_HPP

#include <cstdint>
#include <string>

#include "internal_status.hpp"   // ef::Status

namespace ef {
namespace internal {

class Connection {
public:
    virtual ~Connection() = default;

    // {req,args} JSON round-trip. Writes the response payload into `out`;
    // returns Status (never throws).
    virtual Status request(const std::string& req, const std::string& args,
                           std::string& out) = 0;

    // Protobuf control plane: send `payload` (an encoded ef.v1.Request) as a
    // REQUEST frame; write the response payload bytes into `out` and the frame
    // type into out_type. Returns Status (never throws).
    virtual Status request_raw(const std::string& payload, std::string& out,
                               uint8_t* out_type) = 0;

    virtual bool is_open() const = 0;
    virtual void close()         = 0;
};

}  // namespace internal
}  // namespace ef

#endif  // EF_CONNECTION_HPP
