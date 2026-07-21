////////////////////////////////////////////////////////////////////////////////
//
// File:      mat.cpp
// Purpose:   ef::Mat image container methods (alloc/free/copyTo/getPtr) plus
//            the in-place 180-degree flip used for an upside-down camera.
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

#include <algorithm>
#include <climits>
#include <cstdint>

#include "ef/Core.hpp"

namespace ef {

namespace {

// Bytes per row (plane 0) for a packed/planar layout of the given width. Size
// math is in size_t so a large width can't overflow a signed int mid-multiply.
size_t bytes_per_row(MAT_TYPE t, size_t w) {
    switch (t) {
        case MAT_TYPE::U8_C1: return w;
        case MAT_TYPE::U8_C3: return w * 3;
        case MAT_TYPE::U8_C4: return w * 4;
        case MAT_TYPE::NV12:  return w;      // Y plane row; chroma is a half-size plane
    }
    return 0;
}

// Total buffer size in bytes for (w x h) of the given layout.
size_t buffer_size(MAT_TYPE t, size_t w, size_t h) {
    size_t row = bytes_per_row(t, w);
    // NV12 = full Y plane + half-res interleaved UV -> 1.5x the Y-plane size.
    return t == MAT_TYPE::NV12 ? row * h * 3 / 2 : row * h;
}

// Canonical VIEW for an allocated MAT_TYPE (packed C3/C4 default to BGR-order).
VIEW view_of(MAT_TYPE t) {
    switch (t) {
        case MAT_TYPE::U8_C1: return VIEW::GRAY;
        case MAT_TYPE::U8_C3: return VIEW::BGR;
        case MAT_TYPE::U8_C4: return VIEW::BGRA;
        case MAT_TYPE::NV12:  return VIEW::NV12;
    }
    return VIEW::BGR;
}

// Packed formats (interleaved channels): reverse rows and pixels-within-rows.
void rotate180_packed(uint8_t* d, int w, int h, int ch, int step) {
    auto swap_px = [ch](uint8_t* a, uint8_t* b) {
        for (int c = 0; c < ch; ++c) std::swap(a[c], b[c]);
    };
    for (int y = 0; y < h / 2; ++y) {
        uint8_t* ra = d + (size_t)y * step;
        uint8_t* rb = d + (size_t)(h - 1 - y) * step;
        for (int x = 0; x < w; ++x)
            swap_px(ra + (size_t)x * ch, rb + (size_t)(w - 1 - x) * ch);
    }
    if (h & 1) {
        uint8_t* row = d + (size_t)(h / 2) * step;
        for (int x = 0; x < w / 2; ++x)
            swap_px(row + (size_t)x * ch, row + (size_t)(w - 1 - x) * ch);
    }
}

// NV12: full Y-plane reversal; the interleaved UV plane rotates by reversing
// the order of its (U,V) byte pairs (U/V stay in order within a pair).
void rotate180_nv12(uint8_t* d, int w, int h) {
    size_t ysz = (size_t)w * h;
    std::reverse(d, d + ysz);
    uint8_t* uv = d + ysz;
    size_t   pairs = ysz / 4;   // (w/2)*(h/2) chroma sites, 2 bytes each
    for (size_t i = 0; i < pairs / 2; ++i) {
        uint8_t* a = uv + 2 * i;
        uint8_t* b = uv + 2 * (pairs - 1 - i);
        std::swap(a[0], b[0]);
        std::swap(a[1], b[1]);
    }
}

// Channels in a packed VIEW (0 for the planar NV12, handled separately).
int channels_of(VIEW v) {
    switch (v) {
        case VIEW::GRAY: return 1;
        case VIEW::BGR:
        case VIEW::RGB:  return 3;
        case VIEW::BGRA:
        case VIEW::RGBA: return 4;
        case VIEW::NV12: return 0;
    }
    return 0;
}

}  // namespace

ERROR_CODE Mat::alloc(int width, int height, MAT_TYPE type, MEM mem) {
    if (mem != MEM::CPU)          return ERROR_CODE::UNSUPPORTED;            // GPU not supported
    if (width <= 0 || height <= 0) return ERROR_CODE::INVALID_FUNCTION_CALL;
    // NV12 has a half-resolution chroma plane, so odd dimensions can't tile it.
    if (type == MAT_TYPE::NV12 && ((width & 1) || (height & 1)))
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    // Reject dimensions whose byte counts would overflow int (step_) or blow up
    // the allocation, using the widest layout (C4) as the bound (all in 64-bit).
    const uint64_t w = (uint64_t)width, h = (uint64_t)height;
    const uint64_t row   = w * 4;
    const uint64_t bytes = (type == MAT_TYPE::NV12) ? (w * h * 3 / 2) : (row * h);
    if (row > (uint64_t)INT_MAX || bytes > (uint64_t)INT_MAX)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    resolution_ = {width, height};
    type_       = type;
    view_       = view_of(type);
    memory_     = MEM::CPU;
    step_       = (int)bytes_per_row(type, (size_t)width);
    data_.assign(buffer_size(type, (size_t)width, (size_t)height), 0);
    return ERROR_CODE::SUCCESS;
}

void Mat::free() {
    data_.clear();
    data_.shrink_to_fit();
    step_ = 0;
    resolution_ = {0, 0};
}

ERROR_CODE Mat::copyTo(Mat& dst, MEM mem) const {
    if (mem != MEM::CPU) return ERROR_CODE::UNSUPPORTED;  // GPU not supported
    dst = *this;   // std::vector deep-copies the pixels; metadata copies by value
    return ERROR_CODE::SUCCESS;
}

uint8_t* Mat::getPtr(MEM mem) {
    return mem == MEM::CPU ? data_.data() : nullptr;
}

const uint8_t* Mat::getPtr(MEM mem) const {
    return mem == MEM::CPU ? data_.data() : nullptr;
}

void Mat::flip180() {
    if (data_.empty()) return;
    if (view_ == VIEW::NV12)
        rotate180_nv12(data_.data(), resolution_.width, resolution_.height);
    else
        rotate180_packed(data_.data(), resolution_.width, resolution_.height,
                         channels_of(view_), step_);
}

}  // namespace ef
