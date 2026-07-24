////////////////////////////////////////////////////////////////////////////////
//
// File:      03_calibrate_imu.cpp
// Purpose:   Host-side IMU field calibration, the IMU analog of
//            02_calibrate_camera.cpp. Captures IMU over USB/BLE while the operator
//            (1) holds the device still (gyro zero-bias) and (2) slowly rotates it
//            through all orientations (accel ellipsoid), solves with the pure-C
//            imu_calib routines (which reuse the camera cal's linalg), and writes
//            the result to the device with set_imu_calibration(). Enable it for a
//            session with: ef calibration --imu --mode calibrated.
//
// Copyright (c) 2026, Remnant Robotics, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////
#include <ef/Device.hpp>
#include <ef/ImuCalib.h>     // imc_gyro_bias, imc_accel_ellipsoid (pure C solver)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ef;

// The ellipsoid fit normalizes accel to g, so its offset b comes back in g. The
// device applies a* = M*(x - b) in m/s^2, so bias is scaled back by g and the
// (dimensionless) scale-misalignment W is used as-is. Match the fit's constant.
static constexpr double kGravity = 9.8;
// Field-cal quality gate thresholds (tunable). A poor calibration silently written to
// 100 units is worse than asking the operator to re-run; the reject messages guide them.
static constexpr double kGyroStillMax   = 0.05;  // rad/s per-axis std during the still step
static constexpr double kCoverageMin    = 0.6;   // each accel axis must reach +/- this (unit g)
static constexpr double kAccelFitRmsMax = 0.06;  // RMS(|calibrated|-1) over the tumble, in g

// Capture LIVE IMU for `secs` wall-clock. The device streams IMU into a buffer
// continuously, so we first DRAIN the accumulated backlog (~1.5 s of grab/retrieve,
// discarded), otherwise the first reads return stale samples from before the
// motion and a fixed sample-count capture fills instantly from that backlog (all
// one orientation -> the ellipsoid fit is singular). Then accumulate accel (m/s^2)
// + gyro (rad/s) as interleaved x,y,z over the window, printing a countdown so the
// operator keeps moving. Returns 0 on success, -1 on a link error.
static int collect_imu(Device& device, double secs, const char* what,
                       std::vector<double>& accel, std::vector<double>& gyro) {
    using Clock = std::chrono::steady_clock;
    accel.clear();
    gyro.clear();
    for (auto stop = Clock::now() + std::chrono::milliseconds(1500); Clock::now() < stop;) {
        if (device.grab() != ERROR_CODE::SUCCESS) continue;
        SensorsData drop;
        device.retrieve_imu(drop);                 // discard the stale backlog
    }
    auto end = Clock::now() + std::chrono::milliseconds(static_cast<long long>(secs * 1000));
    int last_rem = -1;
    while (Clock::now() < end) {
        ERROR_CODE g = device.grab();
        if (g == ERROR_CODE::GRAB_TIMEOUT) continue;
        if (g != ERROR_CODE::SUCCESS) {
            std::cerr << "grab failed: " << to_string(g) << "\n";
            return -1;
        }
        SensorsData s;
        if (device.retrieve_imu(s) != ERROR_CODE::SUCCESS) continue;
        for (const ImuSample& smp : s.samples) {
            accel.push_back(smp.acceleration[0]);
            accel.push_back(smp.acceleration[1]);
            accel.push_back(smp.acceleration[2]);
            gyro.push_back(smp.angular_velocity[0]);
            gyro.push_back(smp.angular_velocity[1]);
            gyro.push_back(smp.angular_velocity[2]);
        }
        int rem = static_cast<int>(std::chrono::duration<double>(end - Clock::now()).count()) + 1;
        if (rem != last_rem) {
            std::printf("\r  %s: %2d s left  (%zu samples)   ", what, rem, accel.size() / 3);
            std::fflush(stdout);
            last_rem = rem;
        }
    }
    std::printf("\n");
    return 0;
}

static void wait_enter(const char* msg) {
    std::printf("%s\nPress [ENTER] to start...", msg);
    std::fflush(stdout);
    int c;
    while ((c = std::getchar()) != '\n' && c != EOF) { }
}

int main(int argc, char** argv) {
    InitParameters init;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ble") == 0 && i + 1 < argc) {
            init.input_type = INPUT_TYPE::STREAM;
            init.ble_address = argv[++i];
        } else if (std::strcmp(argv[i], "--password") == 0 && i + 1 < argc) {
            init.ble_password = argv[++i];
        }
    }

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    ImuCalibrationParameters cal;   // identity scale / zero bias defaults

    // ---- Step 1: gyro zero-bias (device still) ----
    wait_enter("Step 1/2 - gyro zero-bias: put the device on a flat, still surface.");
    std::vector<double> a1, g1;
    if (collect_imu(device, 5.0, "still  ", a1, g1) != 0) return 1;
    double gb[3];
    imc_gyro_bias(g1.data(), static_cast<int>(g1.size() / 3), gb);
    cal.gyro_bias = { gb[0], gb[1], gb[2] };
    std::printf("  gyro bias [rad/s] = % .6f % .6f % .6f\n", gb[0], gb[1], gb[2]);
    // Quality gate: the device must be motionless; a moving capture biases the mean.
    {
        const int n = static_cast<int>(g1.size() / 3);
        if (n == 0) {
            std::cerr << "REJECT: no IMU samples captured during the gyro step "
                         "(is the IMU streaming?). Re-run.\n";
            return 2;
        }
        double var[3] = {0, 0, 0};
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) { double d = g1[3*i+k] - gb[k]; var[k] += d*d; }
        double sx = std::sqrt(var[0]/n), sy = std::sqrt(var[1]/n), sz = std::sqrt(var[2]/n);
        std::printf("  gyro stillness [rad/s std] = %.4f %.4f %.4f (limit %.3f)\n",
                    sx, sy, sz, kGyroStillMax);
        if (sx > kGyroStillMax || sy > kGyroStillMax || sz > kGyroStillMax) {
            std::cerr << "REJECT: device was not still during the gyro step. "
                         "Place it on a flat, motionless surface and re-run.\n";
            return 2;
        }
    }

    // ---- Step 2: accel ellipsoid (rotate through all orientations) ----
    wait_enter("Step 2/2 - accel ellipsoid: slowly rotate the device through every\n"
               "orientation (all faces up/down, edges, corners) for ~30 s.");
    std::vector<double> a2, g2;
    if (collect_imu(device, 25.0, "rotate ", a2, g2) != 0) return 1;
    double W[9], b[3];
    if (imc_accel_ellipsoid(a2.data(), static_cast<int>(a2.size() / 3),
                            kGravity, W, b) != 0) {
        std::cerr << "accel ellipsoid fit failed (not enough orientation "
                     "coverage?). Re-run and rotate through all faces.\n";
        return 1;
    }
    for (int i = 0; i < 9; ++i) cal.accel_scale_misalign[i] = W[i];
    for (int i = 0; i < 3; ++i) cal.accel_bias[i] = kGravity * b[i];  // g -> m/s^2
    std::printf("  accel bias [m/s^2] = % .5f % .5f % .5f\n",
                cal.accel_bias[0], cal.accel_bias[1], cal.accel_bias[2]);
    // Quality gate: the tumble must cover all orientations; poor coverage biases the
    // scale (the classic "gravity came out ~9.1" failure), and the calibrated samples
    // must land on the unit sphere. Both are computed from the captured samples + the fit.
    {
        const int n = static_cast<int>(a2.size() / 3);
        double mn[3] = {1e9, 1e9, 1e9}, mx[3] = {-1e9, -1e9, -1e9};
        double sse = 0; int nf = 0;
        for (int i = 0; i < n; ++i) {
            double x = a2[3*i], y = a2[3*i+1], z = a2[3*i+2];
            double m = std::sqrt(x*x + y*y + z*z);
            if (m > 1e-6) {
                double d[3] = { x/m, y/m, z/m };
                for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], d[k]); mx[k] = std::max(mx[k], d[k]); }
            }
            // calibrated magnitude should be ~1 g:  W * (x/g - b)
            double u = x/kGravity - b[0], v = y/kGravity - b[1], w = z/kGravity - b[2];
            double cx = W[0]*u + W[1]*v + W[2]*w;
            double cy = W[3]*u + W[4]*v + W[5]*w;
            double cz = W[6]*u + W[7]*v + W[8]*w;
            double cm = std::sqrt(cx*cx + cy*cy + cz*cz);
            sse += (cm - 1.0) * (cm - 1.0); ++nf;
        }
        double rms = nf ? std::sqrt(sse / nf) : 1.0;
        std::printf("  accel fit RMS [g] = %.4f (limit %.2f)   direction span:"
                    " x[% .2f,% .2f] y[% .2f,% .2f] z[% .2f,% .2f] (need +/-%.1f)\n",
                    rms, kAccelFitRmsMax, mn[0],mx[0], mn[1],mx[1], mn[2],mx[2], kCoverageMin);
        bool covered = mx[0] > kCoverageMin && mn[0] < -kCoverageMin &&
                       mx[1] > kCoverageMin && mn[1] < -kCoverageMin &&
                       mx[2] > kCoverageMin && mn[2] < -kCoverageMin;
        if (!covered) {
            std::cerr << "REJECT: insufficient orientation coverage, each axis must reach +/-"
                      << std::fixed << std::setprecision(1) << kCoverageMin
                      << " g. Rotate through ALL faces, edges and corners, then re-run.\n";
            return 2;
        }
        if (rms > kAccelFitRmsMax) {
            std::cerr << "REJECT: accel fit RMS " << std::fixed << std::setprecision(4) << rms
                      << " g exceeds " << std::setprecision(2) << kAccelFitRmsMax
                      << " (noisy, or the device moved too fast during the tumble). "
                         "Rotate slowly and re-run.\n";
            return 2;
        }
    }

    // gyro_scale_misalign stays identity (bias-only model); noise/tau are left at
    // zero here; they come from the separate Allan-variance characterization, not
    // this field procedure. imu_to_camera / time_offset_ns come from the HW-synced
    // extrinsics (identity + 0 by default).

    // set_imu_calibration is IDLE-only, but grab()/retrieve_imu above put the device
    // into STREAMING and there is no stream-stop verb, so drop and reopen the
    // connection (open() without a grab() leaves the device IDLE) before writing.
    device.close();
    ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "reopen for the calibration write failed: " << to_string(ec) << "\n";
        return 1;
    }

    ec = device.set_imu_calibration(cal);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "set_imu_calibration failed: " << to_string(ec) << "\n";
        return 1;
    }
    std::printf("\nIMU calibration written to the device (applied next session).\n"
                "Record calibrated samples with: ef calibration --imu --mode calibrated\n");
    return 0;
}
