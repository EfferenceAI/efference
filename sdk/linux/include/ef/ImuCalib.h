/* IMU field calibration in pure C: gyro zero-bias and accelerometer
 * ellipsoid fit.
 *
 * Pure C99; the prebuilt libimu_cal.so depends only on libm.
 */
#ifndef IMU_CALIB_H
#define IMU_CALIB_H

#ifdef __cplusplus
extern "C" {
#endif

/* Build/version sanity check for a ctypes/loader. */
int imc_version(void);

/* Gyro zero-bias */
void imc_gyro_bias(const double* g, int n, double bias[3]);

/* Accelerometer ellipsoid fit */
int imc_accel_ellipsoid(const double* a, int n, double gravity,
                        double W[9], double b[3]);

#ifdef __cplusplus
}
#endif
#endif /* IMU_CALIB_H */
