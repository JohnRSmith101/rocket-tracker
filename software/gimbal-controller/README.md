# Gimbal Controller (Arduino Due)

Two sketches. Each is in its own folder because the Arduino IDE requires the folder name to match the sketch name.

| Sketch | Purpose |
|---|---|
| `Rocket_Tracker_2026_R14/` | The gimbal controller. Reads the joystick, receives target position and velocity from the Raspberry Pi, and runs closed-loop PID control of the azimuth and elevation stepper drivers (A4988). |
| `Rocket_Tracker_2026_R14_Logging/` | The same controller plus RAM logging of PID data for performance tests. Instructions are in the folder. |

## Requirements

- Arduino Due.
- Arduino IDE 1.8.13 (the version the author used and tested; the author's instructions say not to use other versions for the logging sketch).
- Libraries, installed from the IDE's Library Manager (Sketch > Include Library > Manage Libraries):
  - **QuickPID** (MIT)
  - **movingAvg** (GPL-3.0; see `../../NOTICE`)

The circuit board for the controller is in `../../hardware/kicad/`. The full wiring, tuning and test procedure is in the Build Book.
