<p align="center">
  <img src="assets/header.png" alt="Efference" width="50%">
</p>

<p align="center">
  The open source SDK for the Efference M1.
</p>

<p align="center">
  <a href="https://efference.ai"><u>Website</u></a> ·
  <a href="https://docs.efference.ai/introduction"><u>Documentation</u></a> ·
  <a href="https://docs.efference.ai/api/device"><u>API Reference</u></a> ·
  <a href="https://x.com/EfferenceAI"><u>X</u></a>
</p>

<p align="center">
  <a href="https://github.com/EfferenceAI/efference/stargazers"><img src="https://img.shields.io/github/stars/EfferenceAI/efference?style=flat&logo=github" alt="GitHub stars"></a>
  <a href="https://github.com/EfferenceAI/efference/network/members"><img src="https://img.shields.io/github/forks/EfferenceAI/efference?style=flat&logo=github" alt="GitHub forks"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/EfferenceAI/efference?style=flat" alt="BSD 3-Clause license"></a>
</p>

Efference provides the host-side C++ interface for our devices. Use them as
standalone systems for distributed data collection or connect them to a host
computer for robot perception applications.

## Getting started

The Efference SDK includes the libraries, tools, and examples needed to integrate
an Efference device. See the [documentation](https://docs.efference.ai/introduction)
for complete setup and usage guides, or browse the
[API reference](https://docs.efference.ai/api/device).

The SDK currently targets Linux and requires C++17, CMake 3.16 or newer,
`pkg-config`, `libusb-1.0`, and `libcurl`.

```sh
git clone https://github.com/EfferenceAI/efference.git
cd efference
./build.sh
```

On Debian or Ubuntu, pass `--deps` on the first build to install dependencies
and the udev rule:

```sh
./build.sh --deps
```

With an Efference device connected over USB:

```sh
./sdk/linux/build/ef-cli info
```

Recordings can be AES-256-GCM encrypted at rest under a key the device generates
and hands over exactly once, and the USB control plane can be locked behind the
same password that already gates Bluetooth LE. Both are off on a factory device:

```sh
./sdk/linux/build/ef-cli encryption create   # generates the key and PRINTS IT, once
./sdk/linux/build/ef-cli encryption on
./sdk/linux/build/ef-cli lock on             # USB now needs --password too
```

Encrypted recordings come off the device as `.enc`, so keep that key: reading one
back needs `ef-decrypt`, which the same build produces.

```sh
./sdk/linux/build/ef-decrypt <in.enc> <key-file> <out>   # key: 32 raw bytes or 64 hex chars
```

The build also produces `efference-viewer`, a live video and IMU window. Both it
and `ef-decrypt` build only when their optional dependencies are present, and the
configure output says which were found.

For installation, device setup, transports, examples, and the complete CLI
reference, visit **[docs.efference.ai](https://docs.efference.ai)**.

## License

Efference is available under the [BSD 3-Clause License](LICENSE).
