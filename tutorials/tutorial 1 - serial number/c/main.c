#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <ef/c/device.h>

int main(int argc, char** argv) {
    int device_id = 0;

    ef_device* device = ef_device_create();
    if (device == NULL) {
        printf("Error: Failed to allocate device memory.\n");
        return 1;
    }

    struct ef_init_parameters init_params;
    init_params.camera_fps = 30;
    init_params.resolution = EF_RESOLUTION_FULL;
    init_params.device_id = device_id;
    init_params.camera_image_flip = EF_FLIP_MODE_AUTO;
    init_params.verbose = 0;
    // TODO: IMU (dynamic range, frequency)
    // TODO: eMMC (recording / not recording)
    // TODO: NPU (active / not active)
    // TODO: NPU flashing
    // TODO: NPU output
    // TODO: ISP adjustments

    int state = ef_device_open(device, &init_params);
    if (state != 0) {
        printf("Failed to open device. Error: %d\n", state);
        ef_device_free(device);
        return 1;
    }

    struct ef_device_information device_info;
    int info_state = ef_device_get_information(device, &device_info);

    if (info_state == 0) {
        printf("Hello. This is my serial number: %d.\n", device_info.serial_number);
    } else {
        printf("Failed to retrieve device information. Error: %d\n", info_state);
        ef_device_close(device);
        ef_device_free(device);
        return 1;
    }

    ef_close_device(device);
    ef_device_free(device);
    
    return 0;
}