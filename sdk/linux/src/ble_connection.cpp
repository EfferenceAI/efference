////////////////////////////////////////////////////////////////////////////////
//
// File:      ble_connection.cpp
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

#include "ble_connection.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

#include "proto.hpp"

namespace ef {
namespace internal {

namespace {

// Control Service UUIDs. MUST match the device firmware.
constexpr const char* SVC_UUID  = "ef000100-9e21-4046-96b4-4d11bf968452";
constexpr const char* CMD_UUID  = "ef000101-9e21-4046-96b4-4d11bf968452";
constexpr const char* RESP_UUID = "ef000102-9e21-4046-96b4-4d11bf968452";
constexpr const char* VER_UUID  = "9529fe05-41ea-4524-b69f-83fec701236d";

constexpr const char* BLUEZ_BUS   = "org.bluez";
constexpr const char* ADAPTER_FALLBACK = "/org/bluez/hci0";
constexpr unsigned    CONNECT_MS  = 15000;
constexpr unsigned    REQUEST_MS  = 5000;
constexpr unsigned    DISCOVER_MS = 25000;  // wait for the advert if BlueZ doesn't know it
constexpr int         CONNECT_TRIES = 4;    // Connect retries (races other BLE agents)

// BlueZ caps WriteValue at the 512-byte GATT attribute limit (InvalidValueLength),
// but control frames can be far larger (an OtaPushChunk request is ~7.2 KB), so
// request_raw slices into <=this-size writes; the device reassembles by the header
// len field.
constexpr size_t WRITE_MAX = 512;

// "<adapter>", "AA:BB:CC:DD:EE:FF" -> "<adapter>/dev_AA_BB_CC_DD_EE_FF"
std::string device_path(const std::string& adapter, const std::string& addr) {
    std::string p = adapter + "/dev_";
    for (char c : addr) p += (c == ':') ? '_' : (char)std::toupper((unsigned char)c);
    return p;
}

// First org.bluez.Adapter1 path BlueZ knows about (else hci0 fallback). Resolve it
// rather than hardcode: the sole adapter isn't always hci0 (USB dongles, multi-radio).
std::string first_adapter_path(GDBusConnection* conn) {
    if (!conn) return ADAPTER_FALLBACK;
    GError* e = nullptr;
    GVariant* r = g_dbus_connection_call_sync(
        conn, BLUEZ_BUS, "/", "org.freedesktop.DBus.ObjectManager",
        "GetManagedObjects", nullptr, G_VARIANT_TYPE("(a{oa{sa{sv}}})"),
        G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (!r) { if (e) g_error_free(e); return ADAPTER_FALLBACK; }
    std::string found;
    GVariantIter* objs = nullptr;
    g_variant_get(r, "(a{oa{sa{sv}}})", &objs);
    const gchar* opath = nullptr;
    GVariantIter* ifaces = nullptr;
    while (found.empty() &&
           g_variant_iter_next(objs, "{&oa{sa{sv}}}", &opath, &ifaces)) {
        const gchar* iname = nullptr;
        GVariantIter* props = nullptr;
        while (g_variant_iter_next(ifaces, "{&sa{sv}}", &iname, &props)) {
            if (std::strcmp(iname, "org.bluez.Adapter1") == 0 && found.empty())
                found = opath;
            g_variant_iter_free(props);
        }
        g_variant_iter_free(ifaces);
    }
    g_variant_iter_free(objs);
    g_variant_unref(r);
    return found.empty() ? ADAPTER_FALLBACK : found;
}

GVariant* empty_opts() {
    GVariantBuilder b;
    g_variant_builder_init(&b, G_VARIANT_TYPE("a{sv}"));
    return g_variant_builder_end(&b);
}

// Scoped thread-default push. GDBus latches the thread-default context at
// g_bus_get_sync()/signal_subscribe(), routing Response notifications to the private
// ctx_ that pump_until() iterates. Must not outlive open(): the pop is thread-affine
// (cross-thread pop corrupts GLib's stack), and while pushed it hijacks the caller's
// default context, silently capturing the app's own GIO async work.
struct CtxGuard {
    GMainContext* c;
    explicit CtxGuard(GMainContext* ctx) : c(ctx) { g_main_context_push_thread_default(c); }
    ~CtxGuard() { g_main_context_pop_thread_default(c); }
    CtxGuard(const CtxGuard&)            = delete;
    CtxGuard& operator=(const CtxGuard&) = delete;
};

}  // namespace

void BleConnection::on_response_bytes(const uint8_t* data, size_t len) {
    // Dropping the buffer is all a notification callback can do about overflow:
    // there is no caller to fail. pump_until then times out on its own.
    if (!proto::append_bounded(rx_, data, len) && verbose_)
        fprintf(stderr, "[ble] response buffer overflow (+%zu); resetting\n", len);
}

// PropertiesChanged on the Response characteristic -> append the Value bytes.
static void on_props_changed(GDBusConnection*, const gchar*, const gchar*,
                             const gchar*, const gchar*, GVariant* params,
                             gpointer user_data) {
    auto* self = static_cast<BleConnection*>(user_data);
    const gchar* iface = nullptr;
    GVariant* changed = nullptr;
    g_variant_get(params, "(&s@a{sv}as)", &iface, &changed, nullptr);
    if (iface && std::strcmp(iface, "org.bluez.GattCharacteristic1") == 0) {
        GVariant* val = g_variant_lookup_value(changed, "Value", G_VARIANT_TYPE("ay"));
        if (val) {
            gsize n = 0;
            const guint8* d = (const guint8*)g_variant_get_fixed_array(val, &n, 1);
            if (d && n) self->on_response_bytes(d, n);
            g_variant_unref(val);
        }
    }
    if (changed) g_variant_unref(changed);
}

template <typename Pred>
bool BleConnection::pump_until(Pred pred, unsigned timeout_ms) {
    gint64 deadline = g_get_monotonic_time() + (gint64)timeout_ms * 1000;
    // Periodic timer so g_main_context_iteration() wakes even with no D-Bus signal
    // pending: polled predicates (e.g. ServicesResolved via sync Get) would otherwise
    // block forever, never re-checking.
    GSource* tick = g_timeout_source_new(50);
    g_source_set_callback(tick, +[](gpointer) -> gboolean { return TRUE; },
                          nullptr, nullptr);
    g_source_attach(tick, ctx_);
    bool ok = true;
    while (!pred()) {
        if (g_get_monotonic_time() >= deadline) { ok = false; break; }
        g_main_context_iteration(ctx_, TRUE);  // wakes at least every 50 ms
    }
    g_source_destroy(tick);
    g_source_unref(tick);
    return ok;
}

bool BleConnection::discover_characteristics() {
    GError* e = nullptr;
    GVariant* r = g_dbus_connection_call_sync(
        conn_, BLUEZ_BUS, "/", "org.freedesktop.DBus.ObjectManager",
        "GetManagedObjects", nullptr, G_VARIANT_TYPE("(a{oa{sa{sv}}})"),
        G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (!r) { if (e) g_error_free(e); return false; }

    GVariantIter* objs = nullptr;
    g_variant_get(r, "(a{oa{sa{sv}}})", &objs);
    const gchar* opath = nullptr;
    GVariantIter* ifaces = nullptr;
    while (g_variant_iter_next(objs, "{&oa{sa{sv}}}", &opath, &ifaces)) {
        // only this device's objects
        if (std::strncmp(opath, dev_path_.c_str(), dev_path_.size()) == 0) {
            const gchar* iname = nullptr;
            GVariantIter* props = nullptr;
            while (g_variant_iter_next(ifaces, "{&sa{sv}}", &iname, &props)) {
                if (std::strcmp(iname, "org.bluez.GattCharacteristic1") == 0) {
                    const gchar* pk = nullptr;
                    GVariant* pv = nullptr;
                    char uuid[64] = {0};
                    while (g_variant_iter_next(props, "{&sv}", &pk, &pv)) {
                        if (std::strcmp(pk, "UUID") == 0)
                            g_strlcpy(uuid, g_variant_get_string(pv, nullptr), sizeof uuid);
                        g_variant_unref(pv);
                    }
                    if      (!g_ascii_strcasecmp(uuid, CMD_UUID))  cmd_path_  = opath;
                    else if (!g_ascii_strcasecmp(uuid, RESP_UUID)) resp_path_ = opath;
                    else if (!g_ascii_strcasecmp(uuid, VER_UUID))  ver_path_  = opath;
                }
                g_variant_iter_free(props);
            }
        }
        g_variant_iter_free(ifaces);
    }
    g_variant_iter_free(objs);
    g_variant_unref(r);
    return !cmd_path_.empty() && !resp_path_.empty();
}

// Does BlueZ have a Device1 object for dev_path_? (Get a cheap property; an
// UnknownObject error means the adapter hasn't seen the advert yet.)
bool BleConnection::device_known() const {
    if (!conn_) return false;
    GError* e = nullptr;
    GVariant* r = g_dbus_connection_call_sync(
        conn_, BLUEZ_BUS, dev_path_.c_str(), "org.freedesktop.DBus.Properties",
        "Get", g_variant_new("(ss)", "org.bluez.Device1", "Address"),
        G_VARIANT_TYPE("(v)"), G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (!r) { if (e) g_error_free(e); return false; }
    g_variant_unref(r);
    return true;
}

bool BleConnection::device_connected() const {
    if (!conn_) return false;
    GError* e = nullptr;
    GVariant* r = g_dbus_connection_call_sync(
        conn_, BLUEZ_BUS, dev_path_.c_str(), "org.freedesktop.DBus.Properties",
        "Get", g_variant_new("(ss)", "org.bluez.Device1", "Connected"),
        G_VARIANT_TYPE("(v)"), G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (!r) { if (e) g_error_free(e); return false; }
    GVariant* v = nullptr; gboolean res = FALSE;
    g_variant_get(r, "(v)", &v);
    if (v) { res = g_variant_get_boolean(v); g_variant_unref(v); }
    g_variant_unref(r);
    return res;
}

// Best-effort StartDiscovery/StopDiscovery (errors ignored: the desktop agent may
// already be scanning). Pins Transport="le" first: BlueZ's default "auto" interleaves
// BR/EDR inquiry with the LE scan, making a BLE-only advert slow/unreliable to catch;
// an LE-only filter scans adverts continuously so the M1 shows up promptly.
void BleConnection::set_discovery(bool on) {
    if (!conn_) return;
    GError* e = nullptr;

    if (on) {
        // SetDiscoveryFilter Transport=le. Best-effort: a conflicting filter from
        // another client may fail this, but the StartDiscovery below still runs.
        GVariantBuilder fb;
        g_variant_builder_init(&fb, G_VARIANT_TYPE("a{sv}"));
        g_variant_builder_add(&fb, "{sv}", "Transport", g_variant_new_string("le"));
        GVariant* fr = g_dbus_connection_call_sync(
            conn_, BLUEZ_BUS, adapter_.c_str(), "org.bluez.Adapter1", "SetDiscoveryFilter",
            g_variant_new("(a{sv})", &fb), nullptr,
            G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
        if (fr) g_variant_unref(fr);
        if (e) {
            if (verbose_) fprintf(stderr, "[ble] SetDiscoveryFilter: %s\n", e->message);
            g_error_free(e);
            e = nullptr;
        }
    }

    GVariant* r = g_dbus_connection_call_sync(
        conn_, BLUEZ_BUS, adapter_.c_str(), "org.bluez.Adapter1",
        on ? "StartDiscovery" : "StopDiscovery", nullptr, nullptr,
        G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (r) g_variant_unref(r);
    if (e) {
        if (verbose_) fprintf(stderr, "[ble] %s: %s\n",
                              on ? "StartDiscovery" : "StopDiscovery", e->message);
        g_error_free(e);
    }
}

// Connect with backoff. Already-connected (or an in-progress connect that completes)
// counts as success, so a racing desktop BLE agent doesn't block the open.
bool BleConnection::connect_with_retry() {
    for (int attempt = 0; attempt < CONNECT_TRIES; ++attempt) {
        if (device_connected()) return true;
        GError* e = nullptr;
        GVariant* r = g_dbus_connection_call_sync(
            conn_, BLUEZ_BUS, dev_path_.c_str(), "org.bluez.Device1", "Connect",
            nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, CONNECT_MS, nullptr, &e);
        if (r) { g_variant_unref(r); return true; }

        const char* msg = e ? e->message : "?";
        // BlueZ reports an already-established/racing link as an error; accept it.
        bool benign = e && (g_strstr_len(msg, -1, "Already Connected") ||
                            g_strstr_len(msg, -1, "AlreadyConnected") ||
                            g_strstr_len(msg, -1, "in progress"));
        if (verbose_) fprintf(stderr, "[ble] Connect(%s) try %d/%d: %s\n",
                              dev_path_.c_str(), attempt + 1, CONNECT_TRIES, msg);
        if (e) g_error_free(e);
        if (benign) return true;
        // Let a racing connect settle, then re-check/retry with backoff.
        if (device_connected()) return true;
        auto never = []() -> bool { return false; };
        pump_until(never, (unsigned)(400 * (attempt + 1)));
    }
    return device_connected();
}

Status BleConnection::open(const std::string& address) {
    if (address.empty()) return Status::DEVICE_NOT_FOUND;
    if (is_open())       return Status::BLE_ERROR;   // not idempotent; close() first

    // A reconnect must not inherit the last link's tail.
    rx_.clear();

    // Any failure after latching of characteristic paths begins must leave the
    // handle reporting !is_open(), so a caller can't mistake a half-open handle
    // for a live one. Clear what discovery may have set on every early exit.
    auto fail = [&](Status s) { cmd_path_.clear(); resp_path_.clear(); ver_path_.clear();
                                return s; };

    // Per-phase elapsed time under --verbose, so a slow open can be attributed
    // to a stage.
    const auto t_open0 = std::chrono::steady_clock::now();
    auto mark = [&](const char* phase) {
        if (!verbose_) return;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t_open0).count();
        fprintf(stderr, "[ble] +%5lld ms  %s\n", (long long)ms, phase);
    };

    ctx_ = g_main_context_new();
    // Pushed for open() only (see CtxGuard): the bus connection and Response signal
    // subscription latch ctx_ here; afterwards pump_until() iterates ctx_ directly.
    CtxGuard guard(ctx_);

    GError* e = nullptr;
    conn_ = g_bus_get_sync(G_BUS_TYPE_SYSTEM, nullptr, &e);
    if (!conn_) {
        if (verbose_) fprintf(stderr, "[ble] system bus: %s\n", e ? e->message : "?");
        if (e) g_error_free(e);
        return Status::BLE_ERROR;
    }
    mark("bluetooth service");
    adapter_  = first_adapter_path(conn_);
    dev_path_ = device_path(adapter_, address);
    mark("adapter found");

    // If BlueZ doesn't know the device yet (no prior scan), discover it, so
    // `ef-cli --ble` self-connects cold without a manual `bluetoothctl connect`.
    if (!device_known()) {
        if (verbose_) fprintf(stderr, "[ble] %s unknown; discovering...\n",
                              address.c_str());
        set_discovery(true);
        mark("scanning");
        auto known = [&]() { return device_known(); };
        bool found = pump_until(known, DISCOVER_MS);
        mark(found ? "device found" : "scan timed out");
        set_discovery(false);
        if (!found) {
            if (verbose_) fprintf(stderr, "[ble] device %s not found in scan\n",
                                  address.c_str());
            return Status::DEVICE_NOT_FOUND;
        }
    }

    // Connect with retry/backoff; already-connected counts as success (agent races).
    // Connecting can take a few seconds: emit progress so it does not look hung.
    if (verbose_) fprintf(stderr, "[ble] %s known; connecting (may take a few seconds)...\n",
                          address.c_str());
    // Exhausted connects mean the device WAS advertising but would not link:
    // that is "detected, unavailable", not "not found". Keeping the codes distinct
    // is what lets Device::open retry only the missed-advert case.
    if (!connect_with_retry()) return Status::BLE_ERROR;
    mark("connected");
    if (verbose_) fprintf(stderr, "[ble] link up; resolving GATT services...\n");

    GVariant* r = nullptr;

    // Wait for GATT discovery (ServicesResolved) then latch characteristics.
    auto resolved = [&]() -> bool {
        GError* ge = nullptr;
        GVariant* pr = g_dbus_connection_call_sync(
            conn_, BLUEZ_BUS, dev_path_.c_str(), "org.freedesktop.DBus.Properties",
            "Get", g_variant_new("(ss)", "org.bluez.Device1", "ServicesResolved"),
            G_VARIANT_TYPE("(v)"), G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &ge);
        if (!pr) { if (ge) g_error_free(ge); return false; }
        GVariant* v = nullptr; gboolean res = FALSE;
        g_variant_get(pr, "(v)", &v);
        if (v) { res = g_variant_get_boolean(v); g_variant_unref(v); }
        g_variant_unref(pr);
        return res;
    };
    if (!pump_until(resolved, CONNECT_MS)) return fail(Status::BLE_ERROR);
    mark("services resolved");
    if (!discover_characteristics())       return fail(Status::INTERFACE_NOT_FOUND);
    mark("characteristics found");

    // Subscribe to Response notifications, then StartNotify.
    resp_sub_ = g_dbus_connection_signal_subscribe(
        conn_, BLUEZ_BUS, "org.freedesktop.DBus.Properties", "PropertiesChanged",
        resp_path_.c_str(), nullptr, G_DBUS_SIGNAL_FLAGS_NONE,
        on_props_changed, this, nullptr);
    r = g_dbus_connection_call_sync(
        conn_, BLUEZ_BUS, resp_path_.c_str(), "org.bluez.GattCharacteristic1",
        "StartNotify", nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (!r) {
        if (verbose_) fprintf(stderr, "[ble] StartNotify: %s\n", e ? e->message : "?");
        if (e) g_error_free(e);
        // Drop the notification subscription we just took, so a retried open()
        // does not orphan it (is_open() is false after fail(), so a retry runs).
        if (resp_sub_) { g_dbus_connection_signal_unsubscribe(conn_, resp_sub_); resp_sub_ = 0; }
        return fail(Status::BLE_ERROR);
    }
    g_variant_unref(r);
    if (verbose_) fprintf(stderr, "[ble] connected %s; control service ready\n",
                          address.c_str());
    return Status::SUCCESS;
}

Status BleConnection::request_raw(const std::string& payload, std::string& out,
                                  uint8_t* out_type) {
    out.clear();
    if (!is_open()) return Status::BLE_ERROR;
    if (payload.size() > proto::MAX_PAYLOAD) return Status::BLE_ERROR;

    std::vector<uint8_t> frame(proto::HDR_LEN + payload.size());
    uint32_t corr = ++corr_;
    frame[0] = proto::MAGIC; frame[1] = proto::VERSION; frame[2] = proto::REQUEST; frame[3] = 0;
    proto::put_le32(&frame[4], corr);
    proto::put_le32(&frame[8], (uint32_t)payload.size());
    if (!payload.empty())
        std::memcpy(&frame[proto::HDR_LEN], payload.data(), payload.size());

    // Fresh accumulator for this exchange (single in-flight, like USB). A late reply
    // to a PREVIOUS request may still land; the corr_id match below discards it.
    rx_.clear();

    // WriteValue in <=WRITE_MAX slices: BlueZ rejects writes past the 512-byte GATT
    // limit (InvalidValueLength) and OTA/upload-URL frames are far larger. The device
    // reassembles by the header len field; sequential sync calls keep slices ordered.
    for (size_t off = 0; off < frame.size(); off += WRITE_MAX) {
        size_t n = frame.size() - off;
        if (n > WRITE_MAX) n = WRITE_MAX;
        GVariant* ay = g_variant_new_fixed_array(G_VARIANT_TYPE_BYTE,
                                                 frame.data() + off, n, 1);
        GError* e = nullptr;
        GVariant* r = g_dbus_connection_call_sync(
            conn_, BLUEZ_BUS, cmd_path_.c_str(), "org.bluez.GattCharacteristic1",
            "WriteValue", g_variant_new("(@ay@a{sv})", ay, empty_opts()),
            nullptr, G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
        if (!r) {
            if (verbose_ && e) fprintf(stderr, "[ble] WriteValue: %s\n", e->message);
            if (e) g_error_free(e);
            return Status::BLE_ERROR;
        }
        g_variant_unref(r);
    }

    // Reassemble frames from notifications (header carries len); accept only the
    // RESPONSE/ERROR whose corr_id matches THIS request (same stale-reply/EVENT
    // skipping as the USB path). Non-matching complete frames are dropped up front.
    auto have_match = [&]() -> bool {
        size_t before = rx_.size();
        bool matched = proto::scan_for_reply(rx_, corr) == proto::Scan::MATCH;
        if (verbose_ && rx_.size() != before)
            fprintf(stderr, "[ble] skipped %zu byte(s) of stale/event data\n",
                    before - rx_.size());
        return matched;
    };
    if (!pump_until(have_match, REQUEST_MS))
        return Status::BLE_ERROR;

    uint8_t  type = rx_[2];
    uint32_t plen = proto::get_le32(&rx_[8]);
    if (out_type) *out_type = type;
    out.assign((const char*)rx_.data() + proto::HDR_LEN, plen);
    return Status::SUCCESS;
}

// One-shot LE scan for M1 peripherals: StartDiscovery, wait, then walk ObjectManager
// for Device1 objects advertising the Control Service UUID. Static, no open connection
// (Device::get_device_list). The bluez daemon scans, so sync calls only, no pumping.
std::vector<BleScanEntry> BleConnection::scan(uint32_t scan_ms, int verbose) {
    std::vector<BleScanEntry> found;
    GError* e = nullptr;
    GDBusConnection* conn = g_bus_get_sync(G_BUS_TYPE_SYSTEM, nullptr, &e);
    if (!conn) {
        if (verbose) fprintf(stderr, "[ble] system bus: %s\n", e ? e->message : "?");
        if (e) g_error_free(e);
        return found;
    }

    const std::string adapter = first_adapter_path(conn);

    // Best-effort LE-only discovery (see set_discovery for why the filter).
    {
        GVariantBuilder fb;
        g_variant_builder_init(&fb, G_VARIANT_TYPE("a{sv}"));
        g_variant_builder_add(&fb, "{sv}", "Transport", g_variant_new_string("le"));
        GVariant* fr = g_dbus_connection_call_sync(
            conn, BLUEZ_BUS, adapter.c_str(), "org.bluez.Adapter1", "SetDiscoveryFilter",
            g_variant_new("(a{sv})", &fb), nullptr,
            G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, nullptr);
        if (fr) g_variant_unref(fr);
        GVariant* sr = g_dbus_connection_call_sync(
            conn, BLUEZ_BUS, adapter.c_str(), "org.bluez.Adapter1", "StartDiscovery",
            nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, nullptr);
        if (sr) g_variant_unref(sr);
    }

    g_usleep((gulong)scan_ms * 1000);

    GVariant* r = g_dbus_connection_call_sync(
        conn, BLUEZ_BUS, "/", "org.freedesktop.DBus.ObjectManager",
        "GetManagedObjects", nullptr, G_VARIANT_TYPE("(a{oa{sa{sv}}})"),
        G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, &e);
    if (r) {
        GVariantIter* objs = nullptr;
        g_variant_get(r, "(a{oa{sa{sv}}})", &objs);
        const gchar* opath = nullptr;
        GVariantIter* ifaces = nullptr;
        while (g_variant_iter_next(objs, "{&oa{sa{sv}}}", &opath, &ifaces)) {
            const gchar* iname = nullptr;
            GVariantIter* props = nullptr;
            while (g_variant_iter_next(ifaces, "{&sa{sv}}", &iname, &props)) {
                if (std::strcmp(iname, "org.bluez.Device1") == 0) {
                    BleScanEntry ent;
                    bool ours = false;
                    // Name is what the remote reported; Alias is a local label bluez
                    // caches and the user can override, so it can mask the per-unit
                    // advert. Collect both and prefer Name after the loop, since
                    // property order is not guaranteed.
                    std::string alias;
                    const gchar* pk = nullptr;
                    GVariant* pv = nullptr;
                    while (g_variant_iter_next(props, "{&sv}", &pk, &pv)) {
                        if (!std::strcmp(pk, "Address"))
                            ent.address = g_variant_get_string(pv, nullptr);
                        else if (!std::strcmp(pk, "Name"))
                            ent.name = g_variant_get_string(pv, nullptr);
                        else if (!std::strcmp(pk, "Alias"))
                            alias = g_variant_get_string(pv, nullptr);
                        else if (!std::strcmp(pk, "UUIDs")) {
                            GVariantIter ui;
                            g_variant_iter_init(&ui, pv);
                            const gchar* u = nullptr;
                            while (g_variant_iter_next(&ui, "&s", &u))
                                if (!g_ascii_strcasecmp(u, SVC_UUID)) ours = true;
                        }
                        g_variant_unref(pv);
                    }
                    if (ent.name.empty()) ent.name = alias;
                    if (ours && !ent.address.empty()) found.push_back(std::move(ent));
                }
                g_variant_iter_free(props);
            }
            g_variant_iter_free(ifaces);
        }
        g_variant_iter_free(objs);
        g_variant_unref(r);
    } else {
        if (verbose && e) fprintf(stderr, "[ble] GetManagedObjects: %s\n", e->message);
        if (e) g_error_free(e);
    }

    GVariant* sp = g_dbus_connection_call_sync(
        conn, BLUEZ_BUS, adapter.c_str(), "org.bluez.Adapter1", "StopDiscovery",
        nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, REQUEST_MS, nullptr, nullptr);
    if (sp) g_variant_unref(sp);
    g_object_unref(conn);
    return found;
}

Status BleConnection::request(const std::string& req, const std::string& args,
                             std::string& out) {
    // Legacy JSON-RPC path (kept for the pre-protobuf control surface). New code
    // goes through request_raw() via Device::pb_call().
    std::string payload = "{\"req\":\"" + req + "\",\"args\":" +
                          (args.empty() ? std::string("{}") : args) + "}";
    uint8_t type = 0;
    Status rc = request_raw(payload, out, &type);
    if (rc != Status::SUCCESS)    return rc;
    if (type == proto::ERROR)     return Status::CONFIGURE_REJECTED;
    if (type != proto::RESPONSE)  return Status::BLE_ERROR;
    return Status::SUCCESS;
}

void BleConnection::close() {
    if (resp_sub_ && conn_) { g_dbus_connection_signal_unsubscribe(conn_, resp_sub_); resp_sub_ = 0; }
    if (conn_ && !resp_path_.empty()) {
        // best-effort StopNotify
        g_dbus_connection_call_sync(conn_, BLUEZ_BUS, resp_path_.c_str(),
            "org.bluez.GattCharacteristic1", "StopNotify", nullptr, nullptr,
            G_DBUS_CALL_FLAGS_NONE, 1000, nullptr, nullptr);
    }
    if (conn_) { g_object_unref(conn_); conn_ = nullptr; }
    // ctx_ was only thread-default within open() (CtxGuard), so close() is safe
    // from any thread: nothing to pop here, just drop the reference.
    if (ctx_)  { g_main_context_unref(ctx_); ctx_ = nullptr; }
    cmd_path_.clear(); resp_path_.clear(); ver_path_.clear();
}

}  // namespace internal
}  // namespace ef
