////////////////////////////////////////////////////////////////////////////////
//
// File:      Parameters.hpp
// Purpose:   [PURPOSE]
// Author:    Gianluca Bencomo
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

struct Matrix4x4 {
    std::array<double, 16> m{};
};

struct float2 {
    float x = 0.f;
    float y = 0.f;
    
    float2() = default;
    float2(float value) : x(value), y(value) {} // Allows single scalar initialization
    float2(float _x, float _y) : x(_x), y(_y) {}
};

struct CalibrationParameters {
    // Focal lengths and principal optical centers
    double fx = 0.0, fy = 0.0;
    double cx = 0.0, cy = 0.0;
    
    double xi = 0.0;
    double alpha = 0.0;

    // leave zero for double sphere
    std::array<double, 12> distortion{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
    
    float h_fov = 0.f;
    float v_fov = 0.f;
    float d_fov = 0.f;

    float focal_length_metric = 0.f;

    lens_distortion_model = LENS_DISTORTION_MODEL::DS
};

struct CameraConfiguration {
    CalibrationParameters calibration_parameters; 
    RESOLUTION mode = RESOLUTION::AUTO;
    int fps  = 0;   
};

struct SensorParameters {
    SENSOR_TYPE type = SENSOR_TYPE::ACCELEROMETER;
    float resolution = 0.f;
    int sampling_rate = 0;
    float2 range = 0.f;
    float noise_density = 0.f;
    float random_walk = 0.f;
    SENSORS_UNIT sensor_unit = SENSORS_UNIT::M_SEC_2;

    bool isAvailable() const {
        return sampling_rate > 0;
    }
};

struct SensorsConfiguration {
    Transform camera_imu_transform;     
    SensorParameters accelerometer_parameters; 
    SensorParameters gyroscope_parameters;     

    bool isAvailable() const {
        return accelerometer_parameters.isAvailable() && 
               gyroscope_parameters.isAvailable();
    }
};

struct WirelessConfiguration {
    // TODO
};

struct StorageConfiguration {
    // TODO
};

struct NeuralConfiguration {
    // TODO
};

struct DeviceInformation {
    unsigned int serial_number = 0;
    unsigned int firmware_version = 0;
    MODEL model = MODEL::M1;
    INPUT_TYPE input_type = INPUT_TYPE::USB;

    CameraConfiguration camera;
    SensorsConfiguration sensor;
    WirelessConfiguration wireless;
    StorageConfiguration storage;
    NeuralConfiguration neural;
};

