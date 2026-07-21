////////////////////////////////////////////////////////////////////////////////
//
// File:      opencv.hpp
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

// OpenCV interop (OPTIONAL, header-only). The core SDK (libef) does NOT depend
// on OpenCV; include this header only in code that already links OpenCV. It lets
// you hand a grabbed frame straight to OpenCV for display (cv::imshow) or
// processing.
//
// Ask retrieve_image() for VIEW::BGR (or VIEW::BGRA) and the returned cv::Mat is
// display-ready with no color conversion; the SDK already color-converts on
// the decode path, and BGR is OpenCV's native channel order.
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

// Map an ef::Mat pixel layout to the matching OpenCV type code.
// NV12 is a single 8-bit plane (Y) followed by interleaved UV at half height, so
// it is wrapped as CV_8UC1 with 1.5x the rows (see toCvMat below).
inline int cvTypeOf(MAT_TYPE type) {
    switch (type) {
        case MAT_TYPE::U8_C1: return CV_8UC1;
        case MAT_TYPE::U8_C3: return CV_8UC3;
        case MAT_TYPE::U8_C4: return CV_8UC4;
        case MAT_TYPE::NV12:  return CV_8UC1;
    }
    return CV_8UC1;
}

// Zero-copy cv::Mat view over an ef::Mat: both share the SAME pixel buffer, so
// this is O(1) with no allocation. The returned cv::Mat is valid only while `m`
// (and its buffer) stays alive and unmodified; call .clone() if you need to own
// the pixels beyond the next grab().
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
