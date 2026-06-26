// ef/device.hpp — Efference SDK public C++ API.
//
// C3 surface: open a wired (USB) M1, pull get_device_information() into a typed
// DeviceInformation, close. The transport speaks the SDK Endpoint wire protocol
// (a 12-byte little-endian header + JSON; see the firmware repo's
// project/app/efference-sdk-endpoint/src/proto.h) over a vendor bulk interface
// (USB class FF / subclass EF / protocol 03). Linux host first.
//
// DeviceInformation mirrors the Function Guide (efference/README.md): "exactly
// our config file + additional information". Fields the firmware does not report
// yet are left at their documented defaults; raw_json always holds the full
// reply for forward-compatibility.
#ifndef EF_DEVICE_HPP
#define EF_DEVICE_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ef {

// ---- status / errors -------------------------------------------------------

// Status of a non-throwing call (open/close). Methods that return rich data
// (get_device_information) throw ef::Exception instead.
enum class ERROR_CODE : int {
    SUCCESS             = 0,
    DEVICE_NOT_FOUND    = 1,  // no M1 (39c5:0001) attached
    INTERFACE_NOT_FOUND = 2,  // device present but no FF/EF/03 SDK interface
    ACCESS_DENIED       = 3,  // permissions (add a udev rule, or run privileged)
    CLAIM_FAILED        = 4,  // interface busy (another process holds it)
    USB_ERROR           = 5,  // libusb init / transfer failure
    ALREADY_OPEN        = 6,
    NOT_OPEN            = 7,
};

// Thrown by get_device_information() (and future data calls) on failure.
// Derives from std::exception so the examples' `catch (const std::exception&)`
// works unchanged.
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& what) : std::runtime_error(what) {}
};

// ---- enums (per the Function Guide) ----------------------------------------

enum class MODEL       { UNKNOWN, M1 };
enum class INPUT_TYPE  { UNKNOWN, USB_C, WIFI, BT };

// Provisional set — confirm against firmware as resolutions are finalized.
enum class RESOLUTION  { UNKNOWN, HD480, HD720, HD1080, HD1200 };

enum class CODEC                { UNKNOWN, RAW, MJPEG, H264, H265 };
enum class RATE_CONTROL         { UNKNOWN, CBR, VBR, FQP };
enum class EXPOSURE_MODE        { AUTO, MANUAL };
enum class WHITE_BALANCE_MODE   { AUTO, MANUAL };
enum class GAMMA_MODE           { LINEAR, SRGB, HDR };
enum class NOISE_REDUCTION_MODE { DISABLED, SPATIAL, SPATIOTEMPORAL };
enum class DISTORTION_MODE      { BYPASS, HARDWARE_LUT, SOFTWARE_CAL };

// ---- nested information blocks ---------------------------------------------

// Row-major 4x4 transform.
struct Matrix4x4 {
    std::array<double, 16> m{};
};

struct CameraInformation {
    int        fps        = 0;
    RESOLUTION resolution = RESOLUTION::UNKNOWN;
    int        width      = 0;
    int        height     = 0;

    // Double Sphere camera model intrinsics.
    double fx = 0, fy = 0;   // focal lengths
    double cx = 0, cy = 0;   // principal point
    double xi = 0;           // first sphere parameter
    double alpha = 0;        // second sphere parameter

    // Extrinsics: translation + rotation quaternion.
    double tx = 0, ty = 0, tz = 0;
    double rw = 0, rx = 0, ry = 0, rz = 0;

    EXPOSURE_MODE auto_exposure_mode = EXPOSURE_MODE::AUTO;
    float manual_exposure_time = 0.f;   // microseconds

    float iso_limit = 0.f;              // max gain (1.0x .. 16.0x)

    WHITE_BALANCE_MODE white_balance_mode = WHITE_BALANCE_MODE::AUTO;
    float manual_white_balance_temperature = 0.f;  // Kelvin

    GAMMA_MODE gamma_mode = GAMMA_MODE::LINEAR;

    NOISE_REDUCTION_MODE noise_reduction_mode = NOISE_REDUCTION_MODE::DISABLED;
    float noise_reduction_strength = 0.f;          // 0.0 .. 1.0

    float sharpening_strength = 0.f;               // 0.0 .. 1.0

    DISTORTION_MODE distortion_mode = DISTORTION_MODE::BYPASS;
};

struct EncodingInformation {
    CODEC        codec        = CODEC::UNKNOWN;
    float        bitrate      = 0.f;   // target bitrate
    RATE_CONTROL rate_control = RATE_CONTROL::UNKNOWN;
};

struct ImuCalibration {
    float accel_noise_density = 0.f;
    float gyro_noise_density  = 0.f;
    float accel_random_walk   = 0.f;
    float gyro_random_walk    = 0.f;
    Matrix4x4 device_to_imu_transform;  // optical center -> IMU
};

struct ImuInformation {
    int  accelerometer_range = 0;   // e.g. ±4g / ±8g
    int  gyroscope_range     = 0;   // e.g. ±250 dps
    int  sampling_rate       = 0;   // e.g. 500 / 1000 Hz
    bool enabled             = false;
    ImuCalibration calibration;
};

struct PackerInformation {
    std::string              format;        // e.g. "mcap"
    std::vector<std::string> topics;        // recorded topics
    int                      segments = 0;  // ms per segment (0 = continuous)
};

struct UsbInformation {
    int log_level = 0;   // 0..7
};

struct EmmcInformation {
    bool backup = false;  // record to eMMC as livestream backup
};

struct WifiInformation {
    std::string ssid;
    std::string ip_address;
    int         log_level = 0;  // 0..7
};

struct BtInformation {
    std::string mac_address;
    bool        paired_state = false;
    int         log_level    = 0;  // 0..7
};

// Device-advertised capabilities (from /etc/intrinsics/device.json, generated
// from the firmware's canonical CAPS[] table): the menu of what the hardware
// *can* do, as opposed to the camera/encoding blocks above which are the
// current *selected* config.
//
// `modes` is the authoritative list — each row is a concrete
// width/height/fps/binning combination with a `usable` flag. A row with
// usable==false is *advertised but not selectable* (e.g. 60 fps on the IMX415);
// configure() rejects it. Match a desired (width,height,fps) against a row with
// usable==true, then pick a codec from `codecs`.
struct Capabilities {
    struct Mode {
        int         width   = 0;
        int         height  = 0;
        int         fps     = 0;
        std::string binning;          // "none" | "2x2"
        bool        usable  = false;  // false => advertised but not selectable
    };
    std::vector<std::string> codecs;          // e.g. H264, H265
    std::vector<std::string> pixel_formats;   // e.g. NV12
    std::vector<std::string> containers;      // e.g. MCAP
    std::vector<Mode>        modes;           // every advertised mode (usable flag per row)
};

// ---- the top-level object --------------------------------------------------

struct DeviceInformation {
    int         serial_number        = 0;
    MODEL       model                = MODEL::UNKNOWN;
    std::string model_name;                       // raw model string
    std::string firmware_version;                 // version_str ("1.0")
    int         firmware_version_int = 0;         // version_int (OTA gate)
    INPUT_TYPE  input_type           = INPUT_TYPE::UNKNOWN;

    CameraInformation   camera;
    EncodingInformation encoding;
    ImuInformation      imu;
    PackerInformation   packer;
    UsbInformation      usb;
    EmmcInformation     emmc;
    WifiInformation     wifi;
    BtInformation       bt;
    Capabilities        caps;           // what the hardware can do

    // Additional information beyond the config file: live reachability of the
    // on-device daemons the endpoint fans into.
    bool orchestrator_reachable = false;
    bool capture_reachable      = false;

    std::string raw_json;   // full reply payload (forward-compat / debugging)
};

// ---- configuration (control plane) -----------------------------------------

// A change to the per-session configuration. ADR-0009: configuration is static
// per session and applied at the next session start; the device's Configure is
// IDLE-only (stop any recording first). Only the fields you SET are sent — a
// field left at its default (0 / empty string) keeps the device's current
// value, so a partial Configuration is a partial update.
//
// The camera tuple (width/height/fps/codec) is validated on the device against
// the advertised caps (Capabilities::modes): an absent geometry, an
// advertised-but-unusable row (e.g. 60 fps), or an unsupported codec is rejected
// and nothing changes (see ConfigureResult::reason). Codec / container are
// case-insensitive here (sent lowercase on the wire).
struct Configuration {
    // Camera tuple — all four are checked together against a usable caps mode.
    int         width     = 0;   // e.g. 1920 / 1280 / 640
    int         height    = 0;   // e.g. 1080 / 720 / 480
    int         fps       = 0;   // e.g. 5 / 10 / 15 / 30
    std::string codec;           // "h264" | "h265"
    std::string container;       // "mcap"

    // Recording / session knobs (optional).
    int         segment_seconds = 0;   // MCAP segment length (seconds)
    std::string capture_mode;          // "usb_sdk" | "wifi_collect" | "livestream" | "uvc"
};

// Outcome of configure(). A transport / protocol failure throws ef::Exception;
// a *device rejection* (bad mode, not idle, unknown key) is a normal result with
// applied==false and a populated reason — the caller picks another mode.
struct ConfigureResult {
    bool        applied        = false;  // true => accepted, validated, persisted
    int         config_version = 0;      // device config_version after a successful apply
    std::string reason;                  // on rejection: offending key / "camera:not_usable" /
                                         // "camera:not_advertised" / "not_idle" / ...
    std::string raw_json;                // full reply payload (forward-compat / debugging)
};

// ---- MCAP streaming (data plane) -------------------------------------------

// Why a stream ended. STOPPED is the only non-terminal end (the caller asked to
// stop, or the duration elapsed) and yields a complete .mcap (footer captured).
// The rest are TERMINAL: the SDK stops and reports — it never auto-recovers; the
// user inspects the reason and drives any resume (mcap-streaming.md §7).
enum class STREAM_END : int {
    STOPPED            = 0,  // clean stop — .mcap is complete (footer written)
    PRODUCER_RESTART   = 1,  // capture restarted underneath us; file is a prefix
    DROPPED            = 2,  // device ring overran (consumer lagged); file is a prefix
    HOST_GONE          = 3,  // ep3 transport error / device detached mid-stream
    START_FAILED       = 4,  // start_stream rejected (no USB / busy / not configured)
    NO_STREAM_ENDPOINT = 5,  // firmware exposes no 2nd bulk IN (pre-M0 build)
    ERROR              = 6,  // protocol / IO / unexpected failure
};

struct StreamResult {
    STREAM_END  end      = STREAM_END::ERROR;
    uint64_t    bytes    = 0;   // bytes written to the output file
    uint64_t    dropped  = 0;   // device ring overrun count at end
    uint32_t    restarts = 0;   // producer restarts observed
    std::string detail;         // device-reported last_end string (raw)
};

// Invoked exactly once when a stream ends, just before stream_mcap() returns,
// so an event-driven caller is notified of the reason. The SDK has already
// stopped — it does NOT resume; resuming is the caller's explicit decision.
using StreamEndCallback = std::function<void(const StreamResult&)>;

// Sink for raw MCAP bytes — invoked on the reader thread with each ep3 chunk,
// in order. It MUST be fast (append + return); doing heavy work here stalls the
// reader and risks a device-side ring overrun (§8 — keep persist/decode/view on
// their own threads). Used by the sink overload of stream_mcap (e.g. a live
// viewer feeding a parser/decoder while also archiving).
using StreamSink = std::function<void(const uint8_t* data, size_t len)>;

// ---- device handle ---------------------------------------------------------

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

    // Discover + claim the device. Returns ALREADY_OPEN if called twice.
    ERROR_CODE open(const InitParameters& params = InitParameters{});

    // READ. Frames get_device_information over USB, parses the reply into the
    // struct. Throws ef::Exception on transport / protocol / parse failure.
    DeviceInformation get_device_information();

    // WRITE. Apply a (partial) session configuration. IDLE-only on the device
    // (ADR-0009) — applied at the next session start. Returns the device's
    // verdict: ConfigureResult::applied + config_version on success, or
    // applied==false + reason on rejection (bad mode, not idle, unknown key).
    // Throws ef::Exception only on transport / protocol failure, never on a
    // plain rejection. Validate intended modes against
    // get_device_information().caps.modes (usable==true) to avoid round-trips.
    ConfigureResult configure(const Configuration& cfg);

    // ---- MCAP data plane (M3) ----------------------------------------------
    // Stream a byte-exact MCAP byte stream from the device to `path`, blocking
    // until `should_stop()` returns true, the duration elapses, or the device
    // signals a terminal end (drop / producer restart / host gone).
    //
    // Mechanics: start_stream over the control channel arms capture (usb_sdk
    // mode); the device pushes the MCAP onto its 2nd bulk IN (ep3). This call
    // drains ep3 to `path` at line rate on the calling thread — it is NOT
    // coupled to any viewer (§8); a live viewer must read `path` independently
    // (e.g. tail it), so a slow viewer can never back-pressure the reader into
    // a device-side overrun. On a clean stop it performs the TWO-PHASE stop
    // (finalize → drain the footer off ep3 → teardown) so `path` is a complete,
    // Foxglove-openable .mcap. On a terminal end `path` is a valid truncated
    // PREFIX (no footer) and the reason is reported — the SDK does NOT resume
    // (§7); `on_end` (if set) and the returned StreamResult carry the reason.
    //
    // Does not throw; failures surface as a STREAM_END in the result.
    StreamResult stream_mcap(const std::string& path,
                             const std::function<bool()>& should_stop,
                             const StreamEndCallback& on_end = {});

    // Convenience: stream for `seconds` (<= 0 means until a terminal end).
    StreamResult stream_mcap(const std::string& path, double seconds,
                             const StreamEndCallback& on_end = {});

    // Same as the file overloads, but delivers the raw MCAP bytes to `sink`
    // (on the reader thread, in order) instead of writing a file — for a live
    // consumer (parser / decoder / viewer) that also wants the two-phase stop +
    // terminal semantics. The caller archives too by writing the bytes itself.
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

// ---- enum -> string (for printing / logging) -------------------------------

const char* to_string(ERROR_CODE);
const char* to_string(STREAM_END);
const char* to_string(MODEL);
const char* to_string(INPUT_TYPE);
const char* to_string(RESOLUTION);
const char* to_string(CODEC);
const char* to_string(RATE_CONTROL);
const char* to_string(EXPOSURE_MODE);
const char* to_string(WHITE_BALANCE_MODE);
const char* to_string(GAMMA_MODE);
const char* to_string(NOISE_REDUCTION_MODE);
const char* to_string(DISTORTION_MODE);

}  // namespace ef

#endif  // EF_DEVICE_HPP
