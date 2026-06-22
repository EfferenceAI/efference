import efference.ef as ef

# shows image stream left and orientation right
# right now we are just recieving 

def main():
    device = ef.device()

    init_params = ef.init_parameters()
    # camera setup
    init_params.camera.fps = 30
    init_params.camera.resolution = ef.RESOLUTION.FULL
    init_params.camera.image_flip = ef.FLIP_MODE.AUTO
    # imu setup
    init_params.accel.range = ef.ACCEL_RANGE
    init_params.accel.frequency = ef.ACCEL_FREQ
    init_params.gyro.range = ef.GYRO_RANGE
    init_params.gyro.frequency = ef.GYRO_FREQ
    init_params.device_id = device_id
    init_params.verbose = 0

    status = device.open(init_params)
    if status != ef.ERROR_CODE.SUCCESS:
        print(f"Failed to open device. Error: {status}")
        exit(1)

