---
title: "Health check"
description: "Run the M1's built-in self-test and print every probe."
---

Runs the device's health sweep and prints the overall verdict plus each probe.
Use it as a go/no-go check before recording or streaming.

## Walkthrough

Open over USB as usual, then run the sweep. `health_check()` fills a
`HealthStatus` on the device; the second argument selects the deeper stress tier
(set here by the `--deep` flag):

```cpp
HealthStatus health;
ec = device.health_check(health, deep);
if (ec != ERROR_CODE::SUCCESS) {
    std::cerr << "health check failed: " << to_string(ec) << "\n";
    device.close();
    return 1;
}
```

`HealthStatus::passed` is the summary; `camera` and `imu` report subsystem
availability; `checks` is the list of individual probes, each with a name, a
pass/fail, and an optional detail string:

```cpp
std::cout << "overall=" << (health.passed ? "PASS" : "FAIL")
          << " camera=" << to_string(health.camera)
          << " imu="    << to_string(health.imu)
          << " checks=" << health.checks.size() << "\n";
for (const auto& c : health.checks)
    std::cout << "  [" << (c.passed ? "PASS" : "FAIL") << "] " << c.name
              << (c.detail.empty() ? "" : (" - " + c.detail)) << "\n";
```

Returning `health.passed ? 0 : 1` lets the program drop into scripts and CI.

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/health_check          # quick sweep
./build/health_check --deep   # add the stress tier
```

## Expected output

```text
overall=PASS camera=AVAILABLE imu=AVAILABLE checks=N
  [PASS] <probe name>
  ...
```
