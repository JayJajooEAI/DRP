# DRP

```
development/     where the detectors are built: code, data, outputs
├── impersonation/    screenshot of a social-media profile -> impersonation or not
├── WebPhish/         link + the stored page -> web phish or not
└── PhysicalThreatFacility/   a message -> physical threat (tied to a real place) or not
inference/       where they are served: one server for all three (code + models only)
├── impersonation/
├── WebPhish/
└── PhysicalThreatFacility/
```
Each folder has its own README. Data and run outputs are not in git.
