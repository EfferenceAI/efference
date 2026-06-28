////////////////////////////////////////////////////////////////////////////////
//
// File:      Enums.hpp
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

#ifndef EF_ENUMS_HPP
#define EF_ENUMS_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ef {

enum class MODEL {  
    M1,
};

enum class INPUT_TYPE {  
    USB,
    STREAM, // WiFi or BT
    MCAP, // our standard for recording data
};
enum class RESOLUTION  { 
    HD1200,
    HD1080,
    SVGA,
    AUTO
};

enum class COMPRESSION_MODE { 
    LOSSLESS,
    H264,
    H264_LOSSLESS,
    H265,
    H265_LOSSLESS,
 };

enum class SENSOR_TYPE {
    ACCELEROMETER,
    GYROSCOPE
}

enum class SENSORS_UNIT {
    M_SEC_2,
    DEG_SEC,
    CELSIUS,
    HERTZ
}

enum class LENS_DISTORTION_MODEL {
    DS
}


} // namespace ef

#endif // EF_DEVICE_DEFINITIONS_HPP