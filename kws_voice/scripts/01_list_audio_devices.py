import sounddevice as sd


def main():
    devices = sd.query_devices()
    print("=== Audio input devices ===")
    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        print(
            f"[{index}] {device['name']} | inputs={max_input_channels} | "
            f"default_sr={device['default_samplerate']}"
        )


if __name__ == "__main__":
    main()
