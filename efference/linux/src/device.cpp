#include "ef/device.hpp"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "json.hpp"
#include "usb_transport.hpp"

namespace ef {

namespace {

using detail::Json;

// ---- string -> enum (accept the Function Guide spellings) ------------------

MODEL model_from(const std::string& s) {
    if (s == "M1") return MODEL::M1;
    return MODEL::UNKNOWN;
}

INPUT_TYPE input_type_from(const std::string& s) {
    if (s == "USB_C" || s == "USB-C" || s == "USB") return INPUT_TYPE::USB_C;
    if (s == "WIFI"  || s == "WiFi")                 return INPUT_TYPE::WIFI;
    if (s == "BT"    || s == "Bluetooth")            return INPUT_TYPE::BT;
    return INPUT_TYPE::UNKNOWN;
}

RESOLUTION resolution_from(const std::string& s) {
    if (s == "480p"  || s == "HD480")  return RESOLUTION::HD480;
    if (s == "720p"  || s == "HD720")  return RESOLUTION::HD720;
    if (s == "1080p" || s == "HD1080") return RESOLUTION::HD1080;
    if (s == "1200p" || s == "HD1200") return RESOLUTION::HD1200;
    return RESOLUTION::UNKNOWN;
}

CODEC codec_from(const std::string& s) {
    // Accept both the caps spelling (UPPER) and the wire-config spelling (lower).
    if (s == "RAW"  || s == "raw")     return CODEC::RAW;
    if (s == "mjpeg" || s == "MJPEG")  return CODEC::MJPEG;
    if (s == "H264" || s == "h264")    return CODEC::H264;
    if (s == "H265" || s == "h265")    return CODEC::H265;
    return CODEC::UNKNOWN;
}

RATE_CONTROL rate_control_from(const std::string& s) {
    if (s == "CBR") return RATE_CONTROL::CBR;
    if (s == "VBR") return RATE_CONTROL::VBR;
    if (s == "FQP") return RATE_CONTROL::FQP;
    return RATE_CONTROL::UNKNOWN;
}

EXPOSURE_MODE exposure_from(const std::string& s) {
    return (s == "Manual" || s == "MANUAL") ? EXPOSURE_MODE::MANUAL : EXPOSURE_MODE::AUTO;
}

WHITE_BALANCE_MODE wb_from(const std::string& s) {
    return (s == "Manual" || s == "MANUAL") ? WHITE_BALANCE_MODE::MANUAL : WHITE_BALANCE_MODE::AUTO;
}

GAMMA_MODE gamma_from(const std::string& s) {
    if (s == "sRGB" || s == "SRGB") return GAMMA_MODE::SRGB;
    if (s == "HDR")                 return GAMMA_MODE::HDR;
    return GAMMA_MODE::LINEAR;
}

NOISE_REDUCTION_MODE nr_from(const std::string& s) {
    if (s == "Spatial")        return NOISE_REDUCTION_MODE::SPATIAL;
    if (s == "Spatiotemporal") return NOISE_REDUCTION_MODE::SPATIOTEMPORAL;
    return NOISE_REDUCTION_MODE::DISABLED;
}

DISTORTION_MODE distortion_from(const std::string& s) {
    if (s == "Hardware_LUT") return DISTORTION_MODE::HARDWARE_LUT;
    if (s == "Software_CAL") return DISTORTION_MODE::SOFTWARE_CAL;
    return DISTORTION_MODE::BYPASS;
}

// Map the firmware's stream last_end string onto a STREAM_END. The non-terminal
// states (the stream is healthy or cleanly stopped) all collapse to STOPPED;
// only the genuine abnormal ends are terminal (mcap-streaming.md §7).
STREAM_END stream_end_from(const std::string& s) {
    if (s == "dropped")          return STREAM_END::DROPPED;
    if (s == "producer_restart") return STREAM_END::PRODUCER_RESTART;
    if (s == "host_gone")        return STREAM_END::HOST_GONE;
    if (s == "error")            return STREAM_END::ERROR;
    return STREAM_END::STOPPED;  // none | streaming | finalizing | stopped
}

// ---- nested-block parsers (each is a no-op if the key is absent) ------------

void parse_camera(const Json& c, CameraInformation& cam) {
    if (!c.is_object()) return;
    cam.fps    = c["fps"].as_int(cam.fps);
    if (c["resolution"].is_string()) cam.resolution = resolution_from(c["resolution"].str);
    cam.width  = c["width"].as_int(cam.width);
    cam.height = c["height"].as_int(cam.height);
    cam.fx = c["fx"].as_double(cam.fx); cam.fy = c["fy"].as_double(cam.fy);
    cam.cx = c["cx"].as_double(cam.cx); cam.cy = c["cy"].as_double(cam.cy);
    cam.xi = c["xi"].as_double(cam.xi); cam.alpha = c["alpha"].as_double(cam.alpha);
    cam.tx = c["tx"].as_double(cam.tx); cam.ty = c["ty"].as_double(cam.ty); cam.tz = c["tz"].as_double(cam.tz);
    cam.rw = c["rw"].as_double(cam.rw); cam.rx = c["rx"].as_double(cam.rx);
    cam.ry = c["ry"].as_double(cam.ry); cam.rz = c["rz"].as_double(cam.rz);
    if (c["auto_exposure_mode"].is_string()) cam.auto_exposure_mode = exposure_from(c["auto_exposure_mode"].str);
    cam.manual_exposure_time = (float)c["manual_exposure_time"].as_double(cam.manual_exposure_time);
    cam.iso_limit = (float)c["iso_limit"].as_double(cam.iso_limit);
    if (c["white_balance_mode"].is_string()) cam.white_balance_mode = wb_from(c["white_balance_mode"].str);
    cam.manual_white_balance_temperature =
        (float)c["manual_white_balance_temperature"].as_double(cam.manual_white_balance_temperature);
    if (c["gamma_mode"].is_string()) cam.gamma_mode = gamma_from(c["gamma_mode"].str);
    if (c["noise_reduction_mode"].is_string()) cam.noise_reduction_mode = nr_from(c["noise_reduction_mode"].str);
    cam.noise_reduction_strength = (float)c["noise_reduction_strength"].as_double(cam.noise_reduction_strength);
    cam.sharpening_strength = (float)c["sharpening_strength"].as_double(cam.sharpening_strength);
    if (c["distortion_mode"].is_string()) cam.distortion_mode = distortion_from(c["distortion_mode"].str);
}

void parse_encoding(const Json& e, EncodingInformation& enc) {
    if (!e.is_object()) return;
    if (e["codec"].is_string())        enc.codec        = codec_from(e["codec"].str);
    enc.bitrate = (float)e["bitrate"].as_double(enc.bitrate);
    if (e["rate_control"].is_string()) enc.rate_control = rate_control_from(e["rate_control"].str);
}

void parse_matrix(const Json& a, Matrix4x4& mat) {
    if (!a.is_array()) return;
    for (size_t i = 0; i < mat.m.size() && i < a.arr.size(); i++)
        mat.m[i] = a.arr[i].as_double(0.0);
}

void parse_imu(const Json& i, ImuInformation& imu) {
    if (!i.is_object()) return;
    imu.accelerometer_range = i["accelerometer_range"].as_int(imu.accelerometer_range);
    imu.gyroscope_range     = i["gyroscope_range"].as_int(imu.gyroscope_range);
    imu.sampling_rate       = i["sampling_rate"].as_int(imu.sampling_rate);
    imu.enabled             = i["enabled"].as_bool(imu.enabled);
    const Json& cal = i["calibration"];
    if (cal.is_object()) {
        imu.calibration.accel_noise_density = (float)cal["accel_noise_density"].as_double(imu.calibration.accel_noise_density);
        imu.calibration.gyro_noise_density  = (float)cal["gyro_noise_density"].as_double(imu.calibration.gyro_noise_density);
        imu.calibration.accel_random_walk   = (float)cal["accel_random_walk"].as_double(imu.calibration.accel_random_walk);
        imu.calibration.gyro_random_walk    = (float)cal["gyro_random_walk"].as_double(imu.calibration.gyro_random_walk);
        parse_matrix(cal["device_to_imu_transform"], imu.calibration.device_to_imu_transform);
    }
}

void parse_packer(const Json& p, PackerInformation& pk) {
    if (!p.is_object()) return;
    pk.format   = p["format"].as_string(pk.format);
    pk.segments = p["segments"].as_int(pk.segments);
    const Json& t = p["topics"];
    if (t.is_array())
        for (const auto& e : t.arr)
            if (e.is_string()) pk.topics.push_back(e.str);
}

void parse_wifi(const Json& w, WifiInformation& wf) {
    if (!w.is_object()) return;
    wf.ssid       = w["ssid"].as_string(wf.ssid);
    wf.ip_address = w["ip_address"].as_string(wf.ip_address);
    wf.log_level  = w["log_level"].as_int(wf.log_level);
}

void parse_bt(const Json& b, BtInformation& bt) {
    if (!b.is_object()) return;
    bt.mac_address  = b["mac_address"].as_string(bt.mac_address);
    bt.paired_state = b["paired_state"].as_bool(bt.paired_state);
    bt.log_level    = b["log_level"].as_int(bt.log_level);
}

void parse_capabilities(const Json& c, Capabilities& caps) {
    if (!c.is_object()) return;
    auto strvec = [&](const char* key, std::vector<std::string>& out) {
        const Json& a = c[key];
        if (a.is_array())
            for (const auto& e : a.arr)
                if (e.is_string()) out.push_back(e.str);
    };
    strvec("codecs",        caps.codecs);
    strvec("pixel_formats", caps.pixel_formats);
    strvec("containers",    caps.containers);

    // Enriched schema: an explicit per-mode array (width/height/fps/binning +
    // usable flag), generated from the firmware's CAPS[]. Replaces the older
    // coarse resolutions[] + framerates_fps[].
    const Json& m = c["modes"];
    if (m.is_array())
        for (const auto& e : m.arr) {
            if (!e.is_object()) continue;
            Capabilities::Mode mode;
            mode.width   = e["width"].as_int();
            mode.height  = e["height"].as_int();
            mode.fps     = e["fps"].as_int();
            mode.binning = e["binning"].as_string();
            mode.usable  = e["usable"].as_bool();
            caps.modes.push_back(mode);
        }
}

}  // namespace

struct Device::Impl {
    detail::UsbTransport usb;
    bool                 open = false;
};

Device::Device() : impl_(new Impl) {}
Device::~Device() { close(); }
Device::Device(Device&&) noexcept            = default;
Device& Device::operator=(Device&&) noexcept = default;

ERROR_CODE Device::open(const InitParameters& params) {
    if (impl_->open) return ERROR_CODE::ALREADY_OPEN;
    ERROR_CODE ec = impl_->usb.open(params.device_id, params.verbose);
    if (ec == ERROR_CODE::SUCCESS) impl_->open = true;
    return ec;
}

void Device::close() {
    if (impl_ && impl_->open) {
        impl_->usb.close();
        impl_->open = false;
    }
}

bool Device::is_open() const { return impl_ && impl_->open; }

DeviceInformation Device::get_device_information() {
    if (!impl_->open) throw Exception("device not open");

    std::string payload = impl_->usb.request("get_device_information", "{}");

    Json j;
    try {
        j = Json::parse(payload);
    } catch (const std::exception& e) {
        throw Exception(std::string("failed to parse device information: ") + e.what());
    }
    if (!j.is_object())
        throw Exception("device information reply is not a JSON object");
    if (j.contains("error"))
        throw Exception("device information error: " + payload);

    DeviceInformation di;
    di.raw_json = payload;

    // ---- identity ----
    // serial_number is authoritative from the device; if the firmware doesn't
    // report it, leave it 0 rather than inventing one host-side.
    di.serial_number = j["serial_number"].as_int(0);

    di.model_name           = j["model"].as_string("unknown");
    di.model                = model_from(di.model_name);
    di.firmware_version     = j["version_str"].as_string();
    di.firmware_version_int = j["version_int"].as_int();
    if (j["input_type"].is_string())
        di.input_type = input_type_from(j["input_type"].str);
    else
        di.input_type = INPUT_TYPE::USB_C;  // this transport is wired

    // ---- config blocks (each populates only if the firmware reports it) ----
    parse_camera(j["camera"],     di.camera);
    parse_encoding(j["encoding"], di.encoding);
    parse_imu(j["imu"],           di.imu);
    parse_packer(j["packer"],     di.packer);
    if (j["usb"].is_object())  di.usb.log_level = j["usb"]["log_level"].as_int(di.usb.log_level);
    if (j["emmc"].is_object()) di.emmc.backup   = j["emmc"]["backup"].as_bool(di.emmc.backup);
    parse_wifi(j["wifi"],         di.wifi);
    parse_bt(j["bt"],             di.bt);
    parse_capabilities(j["capabilities"], di.caps);

    // ---- additional info ----
    di.orchestrator_reachable = j["orchestrator_reachable"].as_bool();
    di.capture_reachable      = j["capture_reachable"].as_bool();

    return di;
}

// ---- configuration (control plane) -----------------------------------------

namespace {

std::string to_lower(std::string s) {
    for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

// Build the set_config request body from only the fields the caller set, so an
// absent field keeps the device's current value (partial update). Codec /
// container go lowercase to match the device's enum tokens.
std::string build_config_body(const Configuration& cfg) {
    std::string b = "{";
    const char* sep = "";
    auto add_int = [&](const char* key, int v) {
        if (v <= 0) return;
        b += sep; sep = ",";
        b += '"'; b += key; b += "\":"; b += std::to_string(v);
    };
    auto add_str = [&](const char* key, const std::string& v, bool lower) {
        if (v.empty()) return;
        b += sep; sep = ",";
        b += '"'; b += key; b += "\":\""; b += (lower ? to_lower(v) : v); b += '"';
    };
    add_int("width",           cfg.width);
    add_int("height",          cfg.height);
    add_int("fps",             cfg.fps);
    add_str("codec",           cfg.codec,        true);
    add_str("container",       cfg.container,    true);
    add_int("segment_seconds", cfg.segment_seconds);
    add_str("capture_mode",    cfg.capture_mode, false);
    b += "}";
    return b;
}

}  // namespace

ConfigureResult Device::configure(const Configuration& cfg) {
    if (!impl_->open) throw Exception("device not open");

    std::string body    = build_config_body(cfg);
    std::string payload = impl_->usb.request("set_config", body);

    Json j;
    try {
        j = Json::parse(payload);
    } catch (const std::exception& e) {
        throw Exception(std::string("failed to parse configure reply: ") + e.what());
    }
    if (!j.is_object())
        throw Exception("configure reply is not a JSON object");

    ConfigureResult r;
    r.raw_json = payload;

    // set_config returns the orchestrator's varlink envelope: success
    // {"parameters":{"config_version":N,"applied":true}}, rejection
    // {"error":"...InvalidState","parameters":{"reason":"..."}}. The endpoint's
    // own failure (e.g. unreachable orchestrator) is {"error":"unreachable",...}.
    // Fall back to the top-level object if there is no "parameters" wrapper.
    const Json& p = j["parameters"].is_object() ? j["parameters"] : j;

    if (j.contains("error")) {
        r.applied = false;
        r.reason  = p["reason"].as_string("");
        if (r.reason.empty()) r.reason = j["error"].as_string("error");
        return r;
    }

    r.applied        = p["applied"].as_bool(false);
    r.config_version = p["config_version"].as_int(0);
    if (!r.applied && r.reason.empty()) r.reason = p["reason"].as_string("");
    return r;
}

// ---- MCAP data plane (M3) --------------------------------------------------

namespace {
constexpr int      kStreamBuf      = 64 * 1024;  // == ring slot size
constexpr unsigned kReadTimeoutMs  = 300;        // ep3 poll while streaming
constexpr unsigned kFooterQuietMs  = 400;        // ep3 poll while draining footer
constexpr int      kFooterIdleStop = 4;          // ~1.6s quiet => footer done
constexpr auto     kStatusPoll     = std::chrono::milliseconds(1000);
}  // namespace

// Core: drain ep3 -> `sink` at line rate, with the two-phase clean stop +
// terminal-end semantics. The file overloads wrap this with a fwrite sink.
StreamResult Device::stream_mcap(const StreamSink& sink,
                                 const std::function<bool()>& should_stop,
                                 const StreamEndCallback& on_end) {
    using clock = std::chrono::steady_clock;
    StreamResult r;

    auto finish = [&](STREAM_END end, const std::string& detail) -> StreamResult {
        r.end = end;
        r.detail = detail;
        if (on_end) on_end(r);
        return r;
    };

    if (!impl_->open)              return finish(STREAM_END::ERROR, "device not open");
    if (!impl_->usb.has_stream())
        return finish(STREAM_END::NO_STREAM_ENDPOINT, "firmware exposes no stream endpoint (ep3)");

    // Poll the device's own drain counters on the control channel. Returns false
    // on a transport error; otherwise fills out the mapped end + raw counters.
    auto poll_status = [&](STREAM_END* mapped, std::string* last_end) -> bool {
        std::string reply;
        try { reply = impl_->usb.request("get_stream_status", "{}"); }
        catch (const std::exception&) { return false; }
        Json s;
        try { s = Json::parse(reply); } catch (const std::exception&) { return false; }
        if (!s.is_object()) return false;
        r.dropped  = (uint64_t)s["dropped"].as_int((int)r.dropped);
        r.restarts = (uint32_t)s["restarts"].as_int((int)r.restarts);
        std::string le = s["last_end"].as_string("");
        if (last_end) *last_end = le;
        if (mapped)   *mapped   = stream_end_from(le);
        return true;
    };

    // ---- arm the stream (start_stream drives capture into usb_sdk COLLECT) ----
    std::string sreply;
    try { sreply = impl_->usb.request("start_stream", "{}"); }
    catch (const std::exception& e) {
        return finish(STREAM_END::ERROR, std::string("start_stream failed: ") + e.what());
    }
    {
        Json sj;
        try { sj = Json::parse(sreply); } catch (const std::exception&) {}
        if (!sj.is_object() || !sj["streaming"].as_bool(false))
            return finish(STREAM_END::START_FAILED, sreply);
    }

    std::vector<uint8_t> buf(kStreamBuf);

    // ---- drain ep3 -> sink at line rate (NOT coupled to any viewer, §8) ------
    STREAM_END  end    = STREAM_END::STOPPED;
    std::string detail = "stopped";
    auto last_poll = clock::now();
    for (;;) {
        if (should_stop && should_stop()) { end = STREAM_END::STOPPED; detail = "stopped"; break; }

        int got = 0;
        int rc  = impl_->usb.read_stream(buf.data(), (int)buf.size(), kReadTimeoutMs, &got);
        if (rc != 0) { end = STREAM_END::HOST_GONE; detail = "host_gone"; break; }
        if (got > 0) { if (sink) sink(buf.data(), (size_t)got); r.bytes += (uint64_t)got; }

        // Terminal-condition watch (drop / producer restart) via the control
        // channel — the device stopped its drain and the user must be told.
        if (clock::now() - last_poll >= kStatusPoll) {
            last_poll = clock::now();
            STREAM_END t = STREAM_END::STOPPED;
            std::string le;
            if (poll_status(&t, &le) && t != STREAM_END::STOPPED) { end = t; detail = le; break; }
        }
    }

    const bool clean = (end == STREAM_END::STOPPED);

    // ---- two-phase stop (mcap-streaming.md §footer fix) ---------------------
    // Phase 1: StopCollect — capture finalizes, pushing the MCAP footer+magic
    // into the ring; the device keeps forwarding it out ep3 ("finalizing").
    try { impl_->usb.request("stop_stream", "{}"); } catch (const std::exception&) {}
    if (clean) {
        // Drain the footer tail off ep3 until quiet -> a complete .mcap.
        int idle = 0;
        for (;;) {
            int got = 0;
            int rc  = impl_->usb.read_stream(buf.data(), (int)buf.size(), kFooterQuietMs, &got);
            if (rc != 0) break;
            if (got == 0) { if (++idle >= kFooterIdleStop) break; continue; }
            idle = 0;
            if (sink) sink(buf.data(), (size_t)got);
            r.bytes += (uint64_t)got;
        }
    }
    // Phase 2: teardown — reclaim the device drain / ep3 / ring.
    try { impl_->usb.request("stop_stream", "{}"); } catch (const std::exception&) {}

    return finish(end, detail);
}

StreamResult Device::stream_mcap(const std::string& path,
                                 const std::function<bool()>& should_stop,
                                 const StreamEndCallback& on_end) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        StreamResult r;
        r.end = STREAM_END::ERROR;
        r.detail = "cannot open output file: " + path;
        if (on_end) on_end(r);
        return r;
    }
    // on_end fires AFTER the file is closed (pass {} to the core, call it here).
    StreamResult r = stream_mcap(
        [f](const uint8_t* d, size_t n) { std::fwrite(d, 1, n, f); },
        should_stop, StreamEndCallback{});
    std::fclose(f);
    if (on_end) on_end(r);
    return r;
}

StreamResult Device::stream_mcap(const std::string& path, double seconds,
                                 const StreamEndCallback& on_end) {
    if (seconds <= 0.0)  // stream until a terminal end
        return stream_mcap(path, std::function<bool()>(), on_end);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds((long long)(seconds * 1e6));
    return stream_mcap(path,
                       [deadline]() { return std::chrono::steady_clock::now() >= deadline; },
                       on_end);
}

// ---- enum -> string --------------------------------------------------------

const char* to_string(ERROR_CODE e) {
    switch (e) {
        case ERROR_CODE::SUCCESS:             return "SUCCESS";
        case ERROR_CODE::DEVICE_NOT_FOUND:    return "DEVICE_NOT_FOUND";
        case ERROR_CODE::INTERFACE_NOT_FOUND: return "INTERFACE_NOT_FOUND";
        case ERROR_CODE::ACCESS_DENIED:       return "ACCESS_DENIED";
        case ERROR_CODE::CLAIM_FAILED:        return "CLAIM_FAILED";
        case ERROR_CODE::USB_ERROR:           return "USB_ERROR";
        case ERROR_CODE::ALREADY_OPEN:        return "ALREADY_OPEN";
        case ERROR_CODE::NOT_OPEN:            return "NOT_OPEN";
    }
    return "UNKNOWN";
}

const char* to_string(STREAM_END e) {
    switch (e) {
        case STREAM_END::STOPPED:            return "STOPPED";
        case STREAM_END::PRODUCER_RESTART:   return "PRODUCER_RESTART";
        case STREAM_END::DROPPED:            return "DROPPED";
        case STREAM_END::HOST_GONE:          return "HOST_GONE";
        case STREAM_END::START_FAILED:       return "START_FAILED";
        case STREAM_END::NO_STREAM_ENDPOINT: return "NO_STREAM_ENDPOINT";
        case STREAM_END::ERROR:              return "ERROR";
    }
    return "ERROR";
}

const char* to_string(MODEL m) {
    switch (m) {
        case MODEL::M1:      return "M1";
        case MODEL::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* to_string(INPUT_TYPE t) {
    switch (t) {
        case INPUT_TYPE::USB_C:   return "USB_C";
        case INPUT_TYPE::WIFI:    return "WIFI";
        case INPUT_TYPE::BT:      return "BT";
        case INPUT_TYPE::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* to_string(RESOLUTION r) {
    switch (r) {
        case RESOLUTION::HD480:   return "HD480";
        case RESOLUTION::HD720:   return "HD720";
        case RESOLUTION::HD1080:  return "HD1080";
        case RESOLUTION::HD1200:  return "HD1200";
        case RESOLUTION::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* to_string(CODEC c) {
    switch (c) {
        case CODEC::RAW:     return "RAW";
        case CODEC::MJPEG:   return "MJPEG";
        case CODEC::H264:    return "H264";
        case CODEC::H265:    return "H265";
        case CODEC::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* to_string(RATE_CONTROL rc) {
    switch (rc) {
        case RATE_CONTROL::CBR:     return "CBR";
        case RATE_CONTROL::VBR:     return "VBR";
        case RATE_CONTROL::FQP:     return "FQP";
        case RATE_CONTROL::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* to_string(EXPOSURE_MODE m) {
    return m == EXPOSURE_MODE::MANUAL ? "MANUAL" : "AUTO";
}

const char* to_string(WHITE_BALANCE_MODE m) {
    return m == WHITE_BALANCE_MODE::MANUAL ? "MANUAL" : "AUTO";
}

const char* to_string(GAMMA_MODE m) {
    switch (m) {
        case GAMMA_MODE::LINEAR: return "LINEAR";
        case GAMMA_MODE::SRGB:   return "sRGB";
        case GAMMA_MODE::HDR:    return "HDR";
    }
    return "LINEAR";
}

const char* to_string(NOISE_REDUCTION_MODE m) {
    switch (m) {
        case NOISE_REDUCTION_MODE::DISABLED:       return "DISABLED";
        case NOISE_REDUCTION_MODE::SPATIAL:        return "SPATIAL";
        case NOISE_REDUCTION_MODE::SPATIOTEMPORAL: return "SPATIOTEMPORAL";
    }
    return "DISABLED";
}

const char* to_string(DISTORTION_MODE m) {
    switch (m) {
        case DISTORTION_MODE::BYPASS:       return "BYPASS";
        case DISTORTION_MODE::HARDWARE_LUT: return "HARDWARE_LUT";
        case DISTORTION_MODE::SOFTWARE_CAL: return "SOFTWARE_CAL";
    }
    return "BYPASS";
}

}  // namespace ef
