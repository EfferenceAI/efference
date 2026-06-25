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

// Device-advertised capabilities (from /etc/intrinsics): the menu of what the
// hardware *can* do, as opposed to the camera/encoding blocks above which are
// the current *selected* config.
struct Capabilities {
    struct Resolution {
        std::string name;      // e.g. "1080p"
        std::string binning;   // e.g. "none", "2x2"
    };
    std::vector<std::string> codecs;          // e.g. H264, H265, MJPEG
    std::vector<std::string> pixel_formats;   // e.g. NV12
    std::vector<std::string> containers;      // e.g. MCAP
    std::vector<int>         framerates_fps;  // e.g. 5,10,15,30,60
    std::vector<Resolution>  resolutions;     // e.g. 1080p/none, 480p/2x2
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

    // Release the interface + handle. Safe to call when not open.
    void close();

    bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- enum -> string (for printing / logging) -------------------------------

const char* to_string(ERROR_CODE);
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
