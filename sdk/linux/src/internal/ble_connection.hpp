////////////////////////////////////////////////////////////////////////////////
//
// File:      ble_connection.hpp
// Purpose:   BlueZ BLE-central control connection (internal).
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

#ifndef EF_BLE_CONNECTION_HPP
#define EF_BLE_CONNECTION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <gio/gio.h>

#include "connection.hpp"
#include "internal_status.hpp"

namespace ef {
namespace internal {

// One discovered M1 peripheral (advertising the Efference Control Service).
struct BleScanEntry {
    std::string address;   // "AA:BB:CC:DD:EE:FF"
    std::string name;      // advertised name, "M1-<serial>" per unit
};

class BleConnection : public Connection {
public:
    explicit BleConnection(int verbose) : verbose_(verbose) {}
    ~BleConnection() override { close(); }
    BleConnection(const BleConnection&)            = delete;
    BleConnection& operator=(const BleConnection&) = delete;

    // Connect to the peripheral at `address` ("AA:BB:CC:DD:EE:FF"), resolve the
    // Control Service characteristics, and subscribe to Response notifications.
    Status open(const std::string& address);

    // One-shot LE scan (scan_ms) for peripherals advertising the Efference
    // Control Service. Static: does not need (or touch) an open connection.
    static std::vector<BleScanEntry> scan(uint32_t scan_ms, int verbose);

    void close() override;
    bool is_open() const override { return conn_ != nullptr && cmd_path_.size() > 0; }

    // {req,args} JSON round-trip → response payload in `out`. Returns Status.
    Status request(const std::string& req, const std::string& args,
                   std::string& out) override;

    // Protobuf control plane: frame `payload` as a REQUEST, WriteValue(Command),
    // reassemble the RESPONSE from notifications; payload into `out`, frame type
    // into out_type. Returns Status. This is the path Device::pb_call() uses.
    Status request_raw(const std::string& payload, std::string& out,
                       uint8_t* out_type) override;


    // Called by the Response PropertiesChanged signal handler (internal).
    void on_response_bytes(const uint8_t* data, size_t len);

private:
    // Walk BlueZ ObjectManager and latch the characteristic object paths.
    bool discover_characteristics();
    // Iterate the private main context until `pred` is true or timeout (ms).
    template <typename Pred> bool pump_until(Pred pred, unsigned timeout_ms);

    // Robust connect helpers (self-connect from a cold state; see .cpp).
    bool device_known() const;      // BlueZ has a Device1 object for dev_path_
    bool device_connected() const;  // Device1.Connected == true
    void set_discovery(bool on);    // best-effort adapter StartDiscovery/StopDiscovery
    bool connect_with_retry();      // Connect w/ backoff; already-connected == success

    int             verbose_ = 0;
    GDBusConnection* conn_   = nullptr;
    GMainContext*    ctx_    = nullptr;   // private; pushed thread-default
    std::string      adapter_ = "/org/bluez/hci0";  // resolved at open(); hci0 fallback
    std::string      dev_path_;           // <adapter>/dev_AA_BB_...
    std::string      cmd_path_, resp_path_, ver_path_;
    guint            resp_sub_ = 0;       // PropertiesChanged subscription id
    uint32_t         corr_     = 0;
    std::vector<uint8_t> rx_;             // RESPONSE reassembly accumulator
};

}  // namespace internal
}  // namespace ef

#endif  // EF_BLE_CONNECTION_HPP
