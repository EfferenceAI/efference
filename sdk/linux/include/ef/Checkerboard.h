////////////////////////////////////////////////////////////////////////////////
//
// File:      Checkerboard.h
// Purpose:   Checkerboard detection + wide-angle (Double Sphere) camera
//            calibration. Public API of the prebuilt libcheckerboard binary.
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

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build/version sanity check for loaders and support triage.
int ckb_version(void);

// ---- detection parameters ---------------------------------------------------
typedef struct {
    int   k_blur;          // directional box-filter length (default 25)
    int   nms_window;      // non-maximal suppression window (default 15)
    float threshold;       // detection threshold (default 0.15)
    int   k_smooth;        // response smoothing window (default 9)
    int   normalize;       // contrast-normalize the response (bool, default 1)
    float contrast_floor;  // min local contrast to accept (default 0.10)
} CkbParams;

// Fill p with the recommended defaults.
void ckb_default_params(CkbParams* p);

// ---- checkerboard detection ---------------------------------------------------
// 8-bit luma (Y) plane -> ordered, subpixel-refined inner corners. `stride` is
// bytes per row (0 or W = packed). Writes 2*cols*rows floats (x, y pairs,
// row-major: corner (r, c) lands at index 2*(r*cols + c)) into out_xy.
// Returns 1 if a full board was found, else 0.
int ckb_detect_board(const uint8_t* gray, int W, int H, int stride,
                     int cols, int rows, const CkbParams* p, float* out_xy);

// ---- wide-angle camera calibration (Double Sphere model) ----------------------
// Levenberg-Marquardt fit of shared intrinsics + per-view extrinsics from a
// planar board (Double Sphere projection; valid to 180-degree+ FOV).
// `board` is n_pts*3 planar points (Z = 0, shared by all views); `imgpoints` is
// n_views*n_pts*2 observed pixels in the same order (e.g. from ckb_detect_board).
//
//   intr6     : [fx, fy, cx, cy, xi, alpha]
//   intr_std6 : 1-sigma parameter uncertainties (~1e300 = unconstrained)
//   rms       : RMS reprojection error in pixels (OpenCV convention)
//   extr      : optional (may be NULL); 6*n_views values [rvec(3), tvec(3)]
//
enum {
    CKB_CALIB_FIX_XI_ZERO = 4,  // force xi=0; use the identifiable 5-param DS submodel
};

// `flags` is 0 or CKB_CALIB_FIX_XI_ZERO. Returns 1 on success, 0 on failure.
int ckb_calibrate_double_sphere(const double* board, const double* imgpoints,
                                int n_views, int n_pts, int img_w, int img_h,
                                int flags, double* intr6, double* intr_std6,
                                double* rms, double* extr);

#ifdef __cplusplus
}
#endif
