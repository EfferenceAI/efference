////////////////////////////////////////////////////////////////////////////////
//
// File:      ef/OpenCV.hpp
// Purpose:   Optional zero-copy bridge between ef::Mat and cv::Mat.
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

// OpenCV interop (OPTIONAL, header-only). The core SDK (libef) does NOT depend on
// OpenCV; include this only in code that already links it. Lets you hand a grabbed
// frame straight to OpenCV for display or processing.
//
// Ask retrieve_image() for VIEW::BGR (or BGRA) and the returned cv::Mat is
// display-ready: the SDK color-converts on decode, and BGR is OpenCV's native order.
//
//   ef::Mat frame;
//   dev.retrieve_image(frame, ef::VIEW::BGR);
//   cv::imshow("efference", ef::toCvMat(frame));   // zero-copy view
//
#ifndef EF_OPENCV_HPP
#define EF_OPENCV_HPP

#include <opencv2/core.hpp>

#include <ef/Core.hpp>
#include <ef/Enums.hpp>

namespace ef {

// Map an ef::Mat pixel layout to the matching OpenCV type code. NV12 (Y plane +
// interleaved UV at half height) is wrapped as CV_8UC1 with 1.5x rows (see toCvMat).
inline int cvTypeOf(MAT_TYPE type) {
    switch (type) {
        case MAT_TYPE::U8_C1: return CV_8UC1;
        case MAT_TYPE::U8_C3: return CV_8UC3;
        case MAT_TYPE::U8_C4: return CV_8UC4;
        case MAT_TYPE::NV12:  return CV_8UC1;
    }
    return CV_8UC1;
}

// Zero-copy cv::Mat view over an ef::Mat: both share the SAME buffer (O(1), no
// alloc). Valid only while `m` stays alive and unmodified; .clone() to keep the
// pixels beyond the next grab().
//
// VIEW::BGR  -> CV_8UC3, ready for cv::imshow.
// VIEW::BGRA -> CV_8UC4, ready for cv::imshow.
// VIEW::NV12 -> CV_8UC1 sized (height*3/2 x width); convert for display with
//               cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12).
// An uninitialised ef::Mat yields an empty cv::Mat.
inline cv::Mat toCvMat(Mat& m) {
    if (!m.isInit())
        return cv::Mat();
    int rows = m.getHeight();
    if (m.getDataType() == MAT_TYPE::NV12)
        rows = m.getHeight() * 3 / 2;   // Y plane + interleaved UV
    return cv::Mat(rows, m.getWidth(), cvTypeOf(m.getDataType()),
                   m.getPtr(), static_cast<size_t>(m.getStep()));
}

}  // namespace ef

#endif  // EF_OPENCV_HPP
