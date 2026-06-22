import efference.ef as ef

def main():
    device = ef.device()

    init_params = ef.init_parameters()
    init_params.verbose = 0

    status = device.open(init_params)
    if status != ef.ERROR_CODE.SUCCESS:
        print(f"Failed to open device. Error: {status}")
        exit(1)

    try:
        device_info = device.get_device_information()
        print(f"Hello. This is my serial number: {device_info.serial_number}.")
    except Exception as e:
        print(f"Failed to retrieve device information. Error: {e}")
        exit(1)
    finally:
        device.close()

if __name__ == "__main__":
    main()