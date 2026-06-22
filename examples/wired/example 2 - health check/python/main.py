import time
import efference.ef as ef

# CONSTANTS
# add the values we should hit for each test to PASS

def test_mainboard(device):
    # check SoC
    # check DDR4
    # check eMMC

def test_perception_and_sync(device):
    # probe tests on mipi and imu
    # runs N frames to host computer making sure connection is stable, measure frame drops
    pass

def test_comms(device):
    # first test bt
    # second test wifi
    # third test usb-c
    pass

def main():
    print("==================================")
    print("----- EFFERENCE HEALTH CHECK -----")
    print("==================================\n")
    
    device = ef.Device()
    init_params = ef.InitParameters()
    init_params.verbose = 0
    
    status = device.open(init_params)
    if status != ef.ERROR_CODE.SUCCESS:
        print(f"Failed to open device. Error: {status}")
        exit(1)

    try:
        test_comms(device)
        test_compute(device)
        test_sensors(device)
        
    except Exception as e:
        print(f"Health check failed. Interrupted by: {e}")
        exit(1)
    finally:
        device.close()

if __name__ == "__main__":
    main()