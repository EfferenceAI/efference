<div align="left">

# EFFERENCE SDK

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)]()
[![Build Status](https://img.shields.io/badge/Build-Passing-success.svg)]()

</div>

---

## Overview

This should be one centralized repo that is the reference point for anyone developing on our platform. Embedded firmware is completely closed-source. SDK is completely open source.

We need to configure the download of this repo so people can started really easily. We want a heavy focus on each of use.

First two immediate deliverables to hit with current device:
1. Wired - plug in, get device info, run health check
2. Wireless - power on, connect, get device info, run health check

For communication, the development flow follows
```
get device info (what are we) > run diagnostics (are we good?) > config setup (are we ready to run?) > running different applications (e.g. data collection, streaming, running models, etc.)
```
Near-term, it's important to figure out what device info we want to pull, negotiating the connection, and setting up the config...these are the pre-requistes for running the device. I am focused on figuring out what this looks like.

**Important:** When we build example applications (e.g. the Efference app), we want to EXCLUSIVELY use this repository.



## Why Efference?

## Getting Started

## Examples

### Getting started
- Plug in device, get info
- plug in device, run diagnostics (health check)
- power device, connect and get info wireless
- power device, connect and run health check wireless

### streaming
- plug in device, preview image stream and take in values. Toggle things like H.265 compression, change isp / imu / image settings
- power device, connect to it, stream, toggle things like above

### update
- plug in device, check for update, update device over the computer OR from the device
- power device, check for update, update device over the device

### recording
- plug in device, configure session, record + save locally
- plug in device, configure session, record + send out.

### field calibration
- take through the 3 steps, wired and unwired