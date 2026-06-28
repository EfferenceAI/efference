## Feedback

1. The first 1000 lines of any codebase set the pace for the next 10,000.
2. **ERROR_CODES.** These are not descriptive and well-informed as they should be added on a need basis. I have corrected these.
3. MCAP is catastropic for latency because we cannot use isochronic protocols for USB or UDP. If we are doing Isochronic / UDP, we cannot use a structured data type because one bit flip will corrupt the entire transmission. We need to avoid MCAP for anything where latency matters and we don't want frame drops. We cannot use MCAP over USB because we cannot do bulk transfers, they are too slow. Data will get corruprted over the wire which is fine for image frame bytes but not fine for structured data like MCAP or JSON. 
4. `libusb_bulk_transfer()` is bad. Bulk transfers have zero latency guarantees. They are strictly "best effort." We need Isochronous Endpoints. 
5. Uncompressed YUV 4:2:2 (16-bit) Fits: ~276 MB/s
6. We have 3 endpoints:
    * ep_out_ (Host to Device) - bulk
    * ep_in_ (Device to Host) - bulk
    * ep_stream_ (Device to Host) - isochronous
7. We mimic the same with WiFi via
    * ep_out_ (Host to Device) - tcp
    * ep_in_ (Device to Host) - tcp
    * ep_stream_ (Device to Host) - udp
8. we need to start simple and it doesn't make sense to add complexity where we haven't decided what we are doing on the ISP side yet for configs. I'm removing a lot of the configs for now because we should be adding them incrementally with the right coding practice:
```
enum class RATE_CONTROL         { UNKNOWN, CBR, VBR, FQP };
enum class EXPOSURE_MODE        { AUTO, MANUAL };
enum class WHITE_BALANCE_MODE   { AUTO, MANUAL };
enum class GAMMA_MODE           { LINEAR, SRGB, HDR };
enum class NOISE_REDUCTION_MODE { DISABLED, SPATIAL, SPATIOTEMPORAL };
enum class DISTORTION_MODE      { BYPASS, HARDWARE_LUT, SOFTWARE_CAL };
```
12. `configuration` makes no sense to be there. we know what's supported and what is not. we don't need to store what the firmware tells us is avaialble for the firmware that we wrote.
13. Exceptions are extremely annoying in robotics. We need to focus on functions that only return error codes / void functions. Everything preallocated. If you look at production robotics codebases (like ROS2 core packages) or automotive safety guidelines (like MISRA C++ and AUTOSAR), exceptions are often banned completely. I removed this:
```
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& what) : std::runtime_error(what) {}
};
```