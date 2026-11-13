/* SPDX-License-Identifier: Apache-2.0

Copyright 2026 John R. Smith

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/* ---------------------------------------------------------
  /*
  Rocket_Tracker_2026_R14_Logging
  For Arduino Due
  Uses QuickPID for AutoTrack closed loop control
  Data logging added
  Uses "filtered" (smoothed) target velocities (deg/sec) from RPi Tracking Camera
  Uses unfiltered (raw) target positions from RPi

   Register-Level Tone Control with:
   - Start / Stop
   - Runtime Frequency Change
   - Non-blocking operation

   Hardware:
   TC0 Channel 0 → Tone 1 (Azimuth)
   TC0 Channel 1 → Tone 2 (Elevation)

   CPU Clock = 84 MHz
   Timer Clock = 84 MHz / 32 = 2.625 MHz
*/
/* ---------------------------- Libraries -------------------------------- */
#include <Arduino.h>
#include <movingAvg.h>
#include <QuickPID.h>
/* ---------------------------- Pins -------------------------------- */
// A4988 Stepper Controller Chips
#define AZModePin1 8
#define AZModePin2 9
#define AZModePin3 10

#define ELModePin1 2
#define ELModePin2 3
#define ELModePin3 4

#define StepperEnablePin 7  // Assumes A4988 driver AZ and EL enabled pins are hardwired togther
#define ELDirPin 6
#define AZDirPin 12
#define PIN_TONE1 11  // Azimuth Step Pin for TONE1 stepping
#define PIN_TONE2 5   // Elevation Step Pin for TONE2 stepping

// Joysticks Pins
#define ThumbAZPin A3
#define ThumbELPin A4
#define ThumbTrig 30;  // for future use

#define AZPin A11
#define ELPin A10
#define JoystickAzVCC 16
#define JoystickElVCC 17
#define JoystickTrigger 51
#define INITIAL_FREQ_TONE1 1000  // Hz (Azimuth Step Frequency)
#define INITIAL_FREQ_TONE2 1000  // Hz (Elevation Step Frequency)
#define TIMER_CLOCK (84000000UL / 32UL)

#define READ_DUE_51() ((PIOA->PIO_PDSR & (1u << 14)) != 0)  //Register Direct DigitalRead Pin 51
#define StageHoldPin 34

/* ---------------------------- Globals ------------------------------ */

//For PID
float AZInput, AZOutput, AZsetpoint;
float ELInput, ELOutput, ELsetpoint;

double TrueTargetX = 0;
double TrueTargetY = 0;

// ------------------------------------------------------------
// AZ PID TUNING NOTES
// EL PID values should be 15.8% larger than AZ PID values due to gearring difference.
// ------------------------------------------------------------
float AZPID_Kp = 90.0;
float AZPID_Kd = 1.4;
float AZPID_Ki = 25.0;

float ELPID_Kp = 104.2;
float ELPID_Kd = 1.62;
float ELPID_Ki = 29.0;
//-------------------------------------------------------------------------
float MeasuredLag = 0.65;  // was 0.6 seconds, from time target appears within FOV to displayed on monitor

float AvgTargetVelX = 0.0;  // deg/sec
float AvgTargetVelY = 0.0;
float FAvgTargetVelX = 0.0;  // deg/sec
float FAvgTargetVelY = 0.0;

volatile bool tone1Running = false;
volatile bool tone2Running = false;

bool state;            // state of motors running (from tones):  false = not running
bool SteppersEnabled;  // state of motors enabled/disabled.  false = disabled

#define INITIAL_FREQ_TONE1 1000  // Hz (Azimuth Step Frequency)
#define INITIAL_FREQ_TONE2 2500  // Hz (Elevation Step Frequency)
#define TIMER_CLOCK (84000000UL / 32UL)

int StageHoldValue;  // Value read from StageHoldPin (0 or 1).

// Predictor inputs from Raspberry Pi (deg/sec)
float TargetVxDegS = 0.0;  // Degrees per second
float TargetVyDegS = 0.0;

//Moving Averages (require integer variables)
movingAvg AvgTargetPosX(4);
movingAvg AvgTargetPosY(4);

// Mechanical model
//Variables for Display Screen Size and Center
int ScreenXSize = 960;  // pixels
int ScreenYSize = 540;

int ScreenXCenter = ScreenXSize / 2;
int ScreenYCenter = ScreenYSize / 2;

float FOV_X_DEG = 3.20;  // degrees, Tracking Camera Field of view
float FOV_Y_DEG = 1.81;  //

float AvgTargetVelXPixSec = 0.0;  // Pixels/Sec target velocity in screen coordinates
float AvgTargetVelYPixSec = 0.0;

// AZ motor Step Angle = 1.8 deg.  EL motor Step Angle = 0.9 deg in full stepping mode.
const float STEPS_PER_DEG_AZ = (200.0 * 16.0 * 1440.0) / 360.0;  // 12800.0 steps/deg when 1/16 microstepping.
const float STEPS_PER_DEG_EL = (400.0 * 16.0 * 834.0) / 360.0;   // 14826.7 steps/deg when 1/16 microstepping.

const float ScreenScaleX_DegPix = FOV_X_DEG / ScreenXSize;  //display screen scale, degrees/pixel, azimuth
const float ScreenScaleY_DegPix = FOV_Y_DEG / ScreenYSize;  //display screen scale, degrees/pixel, azimuth

float TgtDeltaAZ = 0.0;
float TgtDeltaEL = 0.0;

int AZStepperMode;  // AZ stepper operating mode
int ELStepperMode;  // EL Stepper operating mode

float AZStepperSpeed = 1.;  // not sure these need to be float.  Non-zero to avoid divide by zero during Tone setting
float ELStepperSpeed = 1.;
float PrevAZSpeed = 1.;
float PrevELSpeed = 1.;

bool CurrentJSTriggerState = HIGH;  //  Joystick Trigger state.  LOW (or 0) = run JoystickTrack.  HIGH (or 1) allows AutoTrack to run.
bool PreviousJSTriggerState = HIGH;

unsigned long prevAZMicros = 0;
unsigned long prevELMicros = 0;

String TrackMode;

int TargetXNow = ScreenXCenter;  //Nominal screen center (AZ)
int TargetYNow = ScreenYCenter;  //Nominal screen center (EL)

int AvgTargetX;
int AvgTargetY;

//Thrustmaster Joystick
int JoystickCenterAZ = 0;  // Joystick center position in AZ
int JoystickCenterEL = 0;  // Joystick center position in EL

int JoystickAZMargin;  //  Dead Zone around Joystick Cnter in AZ direction
int JoystickELMargin;  //  Dead Zone around Joystick Cnter in EL direction

// Define QuickPID control systems.
// REVERSE preserves the effective polarity of the original PID_v1 sketch.
QuickPID AZPID(&AZInput, &AZOutput, &AZsetpoint, AZPID_Kp, AZPID_Ki, AZPID_Kd, QuickPID::Action::reverse);
QuickPID ELPID(&ELInput, &ELOutput, &ELsetpoint, ELPID_Kp, ELPID_Ki, ELPID_Kd, QuickPID::Action::reverse);



/* ======================================================================
   DATA LOGGER
   ----------------------------------------------------------------------
   Design goal: disturb AutoTrack timing as little as practical.
   During tracking this logger ONLY copies already-calculated values into
   a fixed RAM buffer.  It performs no Serial output, no P/I/D getter calls,
   no lroundf(), no constrain(), and no modulo operation in the record path.

   Programming Port commands (Serial, 115200 baud):
     C  clear the RAM log before a test
     D  dump the RAM log as CSV after the laser is OFF / test has stopped

   The Raspberry Pi remains on SerialUSB and is unaffected by these commands.
   ====================================================================== */
struct LightPIDLogRecord {
  uint32_t time_ms;
  float az_error_px;
  float el_error_px;
  float az_target_vel_deg_s;
  float el_target_vel_deg_s;
};

const uint16_t LOG_CAPACITY = 900;
LightPIDLogRecord PIDLog[LOG_CAPACITY];
uint16_t PIDLogHead = 0;
uint16_t PIDLogCount = 0;

void ClearPIDLog() {
  PIDLogHead = 0;
  PIDLogCount = 0;
  Serial.println("PID log cleared.");
}

inline void RecordPIDLogLight(float avgVelX, float avgVelY) {
  LightPIDLogRecord &r = PIDLog[PIDLogHead];
  r.time_ms = millis();
  r.az_target_vel_deg_s = avgVelX;
  r.el_target_vel_deg_s = avgVelY;

  PIDLogHead++;
  if (PIDLogHead >= LOG_CAPACITY) PIDLogHead = 0;
  if (PIDLogCount < LOG_CAPACITY) PIDLogCount++;
}

void DumpPIDLog() {
  Serial.println("BEGIN_LOG");
  Serial.println("time_ms,AvgTargetVelX_deg_s,AvgTargetVelY_deg_s");

  uint16_t idx = (PIDLogCount == LOG_CAPACITY) ? PIDLogHead : 0;
  for (uint16_t i = 0; i < PIDLogCount; i++) {
    const LightPIDLogRecord &r = PIDLog[idx];
    Serial.print(r.time_ms);
    Serial.print(',');
    Serial.print(r.az_target_vel_deg_s, 5);
    Serial.print(',');
    Serial.println(r.el_target_vel_deg_s, 5);

    idx++;
    if (idx >= LOG_CAPACITY) idx = 0;
  }
  Serial.print("END_PID_LOG records=");
  Serial.println(PIDLogCount);
}

void HandlePIDLogCommands() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c == 'C' || c == 'c') {
    ClearPIDLog();
  } else if (c == 'D' || c == 'd') {
    DumpPIDLog();
  }
  while (Serial.available()) Serial.read();
}

// Motor speed Limits
const int MaxStepperSpeedAZ = 15000;  // (steps/sec). ChatGPT recommends 15000 max to avoid interrupt interference
const int MaxStepperSpeedEL = 15000;  // (steps/sec). ChatGPT recommends 15000 max to avoid interrupt interference

//Time measurements
unsigned long LastUpdateTime = 0;              // msec
unsigned long Timeout;                         // msec.  After this period, location data is considered stale and motors are stopped.
const unsigned long TimeoutNoHold = 2000;      // msec.  Value assigned to varialble Timeout for Non-Staging operations
const unsigned long TimeoutStageHold = 11000;  // msec.  Value assigned to vatiable Timeout for Stage Hold operations.

/* ======================================================================
   SETUP
   ====================================================================== */
void setup() {
  pinMode(StepperEnablePin, OUTPUT);

  pinMode(AZDirPin, OUTPUT);
  pinMode(ELDirPin, OUTPUT);

  pinMode(AZModePin1, OUTPUT);
  pinMode(AZModePin2, OUTPUT);
  pinMode(AZModePin3, OUTPUT);
  pinMode(ELModePin1, OUTPUT);
  pinMode(ELModePin2, OUTPUT);
  pinMode(ELModePin3, OUTPUT);

  pinMode(PIN_TONE1, OUTPUT);  //AZ step pin
  pinMode(PIN_TONE2, OUTPUT);  //EL step pin

  digitalWrite(PIN_TONE1, LOW);
  digitalWrite(PIN_TONE2, LOW);

  setupTimerChannel(0);
  setupTimerChannel(1);

  setTone1Frequency(INITIAL_FREQ_TONE1);
  setTone2Frequency(INITIAL_FREQ_TONE2);

  Timeout = TimeoutNoHold;  // Age for "fresh" data and to keep motors running for normal operations

  // Set motors stepping mode
  AZStepperMode16();  // 1/16 stepping, AZ motor
  ELStepperMode16();  // 1/16 stepping, EL motor

  //Initialize moving averages
  AvgTargetPosX.begin();
  AvgTargetPosY.begin();

  pinMode(51, INPUT_PULLUP);
  pinMode(34, INPUT_PULLUP);

  DisableSteppers();     // Disable stepper motors
  Serial.begin(115200);  // Initialize comms with programming computer

  const uint32_t PIN = 51;  // JoystickTrigger Pin assignment
  pinMode(PIN, INPUT_PULLUP);

  // Enable the PIO controller clock for whatever port this JoystickTrigger lives on
  enablePioClock(g_APinDescription[PIN].ulPeripheralId);

  // Ensure PIO (GPIO) control is enabled and direction is input
  g_APinDescription[PIN].pPort->PIO_PER = g_APinDescription[PIN].ulPin;  // PIO enable
  g_APinDescription[PIN].pPort->PIO_ODR = g_APinDescription[PIN].ulPin;  // input enable

  JoystickAZMargin = 100;  // Was 200. "Dead zone" distance from joystick center in AZ direction (measured stdev = 22.7)
  JoystickELMargin = 50;   // Was 200. "Dead zone" distance from joystick center in EL direction (measured stdev = 7.8)

  pinMode(JoystickAzVCC, OUTPUT);
  pinMode(JoystickElVCC, OUTPUT);
  digitalWrite(JoystickAzVCC, HIGH);  // provides power to joystic AZ potentiometer
  digitalWrite(JoystickElVCC, HIGH);  // provides power to joystic EL potentiometer

  SerialUSB.begin(115200);  // Initialize comms with Raspberry Pi.
  SerialUSB.setTimeout(1);  // Set timeout to 1 ms.  May not need this line;  it was added by ChatGPT.

  //PID Library Setup Settings:
  AZPID.SetOutputLimits(-MaxStepperSpeedAZ, MaxStepperSpeedAZ);
  ELPID.SetOutputLimits(-MaxStepperSpeedEL, MaxStepperSpeedEL);

  // Initialize QuickPID. Defaults are proportional-on-error, derivative-on-measurement,
  // and conditional anti-windup. 100 ms sample time matches PID_v1 default.
  AZPID.SetSampleTimeUs(100000);
  ELPID.SetSampleTimeUs(100000);
  AZPID.SetAntiWindupMode(QuickPID::iAwMode::iAwClamp);
  ELPID.SetAntiWindupMode(QuickPID::iAwMode::iAwClamp);
  AZPID.SetMode(QuickPID::Control::automatic);
  ELPID.SetMode(QuickPID::Control::automatic);

  AZsetpoint = ScreenXCenter;
  ELsetpoint = ScreenYCenter;

  while (!SerialUSB) {};  // Wait for Raspberry Pi Serial to come online
  delay(10);

  CenterJoystick();  // Read Thrustmaster Joystick potentiometers in neutral position and set values to center
}

/* ======================================================================
   LOOP
   ====================================================================== */
void loop() {
  //Serial.println("Top");
  PreviousJSTriggerState = CurrentJSTriggerState;
  CurrentJSTriggerState = fastDigitalRead(51);  //Fast digitalRead of JoystickTrigger state
  delay(1);

  if (CurrentJSTriggerState == LOW) {
    JoystickTrack();
    delay(1);
  }

  AutoTrack();  //Updates and runs motors' speeds and directions when new target location data is available

  // Stop tones and disable motors after data becomes stale if motors are not aready disabled and if joystick is not triggered
  if (((millis() - LastUpdateTime) > Timeout) && (SteppersEnabled == true) && (CurrentJSTriggerState == HIGH)) {
    // Serial.println("Execute Standby Functions");

    DisableSteppers();  //Disable motors.  May be commended out for Test Purposes
    stopTone1();        // Stop motor tones
    stopTone2();
    state = false;  // set motors state as not being stepped
    Timeout = TimeoutNoHold;
    TrackMode = "  ";
    //SerialUSB.println(TrackMode);  //Send track mode to Raspberry Pi Comment out for test purposes
  }
  // Check the Programming Port only after normal tracking/timeout work is complete.
  // With no command waiting, this is only a quick Serial.available() test.
  HandlePIDLogCommands();
}

/* ======================================================================
  JoystickTrack Mode
  ====================================================================== */
void JoystickTrack() {
  //Serial.println("Entered Joystick");

  int AZReadNow;
  int ELReadNow;
  int AZReadPrevious;
  int ELReadPrevious;
  float AZReadFiltered = 0;
  float ELReadFiltered = 0;

  TrackMode = "Joystick Tracking";
  //SerialUSB.println(TrackMode);  //Send track mode to Raspberry Pi. Commented out for test purposes

  if (SteppersEnabled != true) {
    EnableSteppers();  // Enable motors if not already enabled
  }
  analogReadResolution(12);

  while (CurrentJSTriggerState == LOW) {
    AZReadNow = analogRead(AZPin);
    ELReadNow = analogRead(ELPin);

    StageHoldValue = fastPinState(StageHoldPin);  // Check if Timeout should be extended for staging

    if (StageHoldValue == LOW) {
      Timeout = TimeoutStageHold;
    } else {
      Timeout = TimeoutNoHold;
    }

    //   delay(1);

    // Low pass filter because joystick readings are noisy
    AZReadFiltered = 0.95 * AZReadFiltered + 0.05 * AZReadNow;
    ELReadFiltered = 0.95 * ELReadFiltered + 0.05 * ELReadNow;
    AZReadPrevious = AZReadNow;
    ELReadPrevious = ELReadNow;
    AZReadNow = AZReadFiltered;
    ELReadNow = ELReadFiltered;

    if (ReadPiCoordinates()) {};  // continue to empty Pi buffer so RPi doesn't lock up
    delay(1);

    /*
      Joystick EL range is from 80 to 3590 (emperial data).
      Joystick AZ range is from 530 to 3670
      Define Joystick EL Slew as being from < 200 or > 3400
      Define Joystick AZ Slew as being from < 700 or > 3500.
    */

    //Set motor directions and speed.  Minimum speeds set >1 to avoid divide by zero in setting tone values
    if ((AZReadNow != AZReadPrevious) || (ELReadNow != ELReadPrevious)) {
      if (ELReadNow < (JoystickCenterEL - JoystickELMargin)) {
        digitalWrite(ELDirPin, HIGH);
        ELStepperSpeed = map(ELReadNow, 200, (JoystickCenterEL - JoystickELMargin),
                             MaxStepperSpeedEL, 2);
      } else if (ELReadNow > (JoystickCenterEL + JoystickELMargin)) {
        digitalWrite(ELDirPin, LOW);
        ELStepperSpeed = map(ELReadNow, (JoystickCenterEL + JoystickELMargin), 3400,
                             2, MaxStepperSpeedEL);
      } else ELStepperSpeed = 2;

      if (AZReadNow > (JoystickCenterAZ + JoystickAZMargin)) {
        digitalWrite(AZDirPin, HIGH);
        AZStepperSpeed = map(AZReadNow, (JoystickCenterAZ + JoystickAZMargin), 3500,
                             2, MaxStepperSpeedAZ);
      } else if (AZReadNow < (JoystickCenterAZ - JoystickAZMargin)) {
        digitalWrite(AZDirPin, LOW);
        AZStepperSpeed = map(AZReadNow, 700, (JoystickCenterAZ - JoystickAZMargin),
                             MaxStepperSpeedAZ, 2);
      } else AZStepperSpeed = 2;

      //Slew motors if joystickis at extreem end of travel
      if ((AZReadNow < 700) || (AZReadNow > 3500)) {
        setTone1Frequency(MaxStepperSpeedAZ);
      } else {
        setTone1Frequency(AZStepperSpeed);
      }

      if ((ELReadNow < 200) || (ELReadNow > 3500)) {
        setTone2Frequency(MaxStepperSpeedEL);
      } else {
        setTone2Frequency(ELStepperSpeed);
      }
    }

    CurrentJSTriggerState = fastDigitalRead(51);  //Check if joystick trigger is still depressed

    if (state == false) {
      startTone1();
      delay(1);
      startTone2();
      delay(1);
      state = true;
    }
  }
  // Prepare to exit JoystickTrack
  TargetXNow = ScreenXCenter;  //Nominal screen center (AZ)
  TargetYNow = ScreenYCenter;  //Nominal screen center (EL)
  AvgTargetX = ScreenXCenter;
  AvgTargetY = ScreenYCenter;
  TrueTargetX = ScreenXCenter;
  TrueTargetY = ScreenYCenter;
  TgtDeltaAZ = 0.;
  TgtDeltaEL = 0.;
  AvgTargetVelX = 0.0;  // deg/sec
  AvgTargetVelY = 0.0;
  FAvgTargetVelX = 0.0;  // deg/sec
  FAvgTargetVelY = 0.0;

  // If no data available from RPi, stop motors
  if (!ReadPiCoordinates()) {
    stopTone1();
    // delay(1);
    stopTone2();
    //delay(1);
    state = false;
  }
  TrackMode = " ";
  //SerialUSB.println(TrackMode);  //Send track mode to Raspberry Pi.  Comment out for test purposes
  delay(1);
  //Serial.println("Exit Joystick");
}


/* ======================================================================
  AutoTrack — PID using Arduino PID Library
  ====================================================================== */
void AutoTrack() {

  // This void updates and starts motors' speeds (steps/sec) & directions using tones to step the motors
  // This void is the "default" void in main Loop.
  //Serial.println("Entered Auto");


  // Update (TargetXNow, TargetYNow, Filtered_TargetVxDegS, Filtered_TargetVyDegS,) if available
  if (ReadPiCoordinates()) {
    if (SteppersEnabled != true) {
      EnableSteppers();  // Enable Steppers if not already enabled
    }

    TrackMode = "AutoTracking";
    //SerialUSB.println(TrackMode);  // Send track mode to Raspberry Pi.  May be commented out for test purposes

    StageHoldValue = fastPinState(StageHoldPin);  // Check if Timeout should be extended for staging
    if (StageHoldValue == LOW) {
      Timeout = TimeoutStageHold;
    } else {
      Timeout = TimeoutNoHold;
    }

    // Use moving averages function for Target position.  Note, Moving average function is integer.
    AvgTargetX = AvgTargetPosX.reading(TargetXNow);
    AvgTargetY = AvgTargetPosY.reading(TargetYNow);

    //Receiving filtered TargetVelVxDegS and filtered TargetVelVyDegS from Raspberry Pi, so no VelX or VelY averaging
    float FAvgTargetVelX = TargetVxDegS; // Target velocities (Deg/sec)
    float FAvgTargetVelY = TargetVyDegS;

    AvgTargetVelXPixSec = FAvgTargetVelX / ScreenScaleX_DegPix;  //Pixels/sec
    AvgTargetVelYPixSec = FAvgTargetVelY / ScreenScaleX_DegPix;

    //For Plotter
    /*
      Serial.print("AvgTargetVelXPixSec:");
      Serial.println(AvgTargetVelXPixSec);
      Serial.print("AvgTargetVelYPixSec:");
      Serial.println(AvgTargetVelYPixSec);
    */
    ThumbStickCorrection();  // update target centerpoint coordinates to center it on display

    TrueTargetX = AvgTargetX + AvgTargetVelXPixSec * MeasuredLag;  // Predicted true position in presence of system lag
    TrueTargetY = AvgTargetY + AvgTargetVelYPixSec * MeasuredLag;

    TgtDeltaAZ = AZsetpoint - TrueTargetX;  // display units (pixels)
    TgtDeltaEL = ELsetpoint - TrueTargetY;  // display units (pixels)

    // Assign values for QuickPID
    AZInput = TrueTargetX;
    ELInput = TrueTargetY;

    // QuickPID runs continuously through center and through sweep reversals.
    AZPID.Compute();
    ELPID.Compute();

    // Treat PID output as a genuinely signed motor command.  Position error
    // does NOT suppress direction or speed for a moving target.

    int8_t AZCommandSign = (AZOutput < 0.0f) ? -1 : ((AZOutput > 0.0f) ? 1 : 0);
    int8_t ELCommandSign = (ELOutput < 0.0f) ? -1 : ((ELOutput > 0.0f) ? 1 : 0);

    if (AZCommandSign != 0) {
      digitalWrite(AZDirPin, (AZOutput < 0.0f) ? LOW : HIGH);
    }
    if (ELCommandSign != 0) {
      digitalWrite(ELDirPin, (ELOutput < 0.0f) ? HIGH : LOW);
    }

    AZStepperSpeed = fabs(AZOutput);
    ELStepperSpeed = fabs(ELOutput);

    AZStepperSpeed = constrain(AZStepperSpeed, 1.0f, (float)MaxStepperSpeedAZ);
    setTone1Frequency(AZStepperSpeed);
    delay(1);

    ELStepperSpeed = constrain(ELStepperSpeed, 1.0f, (float)MaxStepperSpeedEL);
    setTone2Frequency(ELStepperSpeed);
    delay(1);

    // Start motors if required and not already running
    if (((millis() - LastUpdateTime) < Timeout) && (SteppersEnabled == true) && (state == false)) {
      startTone1();
      delay(1);
      startTone2();
      delay(1);
      state = true;
    }
    // Lightweight RAM capture occurs only after all normal R11 motor-control
    // work for this target update has completed.
    RecordPIDLogLight(FAvgTargetVelX, FAvgTargetVelY);  // (deg/sec)

    //for plotter
    //Serial.print("AZInput:");
    //Serial.println(AZInput);
    //Serial.print(",");
    //Serial.print("AZOutput:");
    // Serial.println(AZOutput);
    /*
      Serial.print("AZStepperSpeed:");
      Serial.print(AZStepperSpeed);
      Serial.print(",");
      Serial.print("ELStepperSpeed:");
      Serial.println(ELStepperSpeed);
    */
  }
  //Serial.println("Exit Auto");
}

/* ======================================================================
  Parse Raspberry Pi incoming data:
  Format uses commas: X,Y,VxDegS,VyDegS
  ====================================================================== */

bool ReadPiCoordinates() {
  if (!SerialUSB.available()) return false;

  String PiString = SerialUSB.readStringUntil('\n');
  PiString.trim();
  if (PiString.length() < 3) return false;

  // New format uses commas: X,Y,VxDegS,VyDegS
  if (PiString.indexOf(',') >= 0) {
    int c1 = PiString.indexOf(',');
    int c2 = PiString.indexOf(',', c1 + 1);
    int c3 = PiString.indexOf(',', c2 + 1);
    if (c1 <= 0 || c2 <= c1) return false;

    float x = PiString.substring(0, c1).toFloat();
    float y = PiString.substring(c1 + 1, c2).toFloat();

    float vx = 0.0;
    float vy = 0.0;
    if (c3 > c2) {
      vx = PiString.substring(c2 + 1, c3).toFloat();
      vy = PiString.substring(c3 + 1).toFloat();
    }

    TargetXNow = (int)round(x);
    TargetYNow = (int)round(y);
    TargetVxDegS = vx;
    TargetVyDegS = vy;

    LastUpdateTime = millis();
    return true;
  }
}

/* ======================================================================
  Thumb-stick correction
  ====================================================================== */
void ThumbStickCorrection() {
  analogReadResolution(10);
  int AZ = analogRead(ThumbAZPin);
  int EL = analogRead(ThumbELPin);

  // Decrement or increment seetpoint values depending on Thumbstick positions.  This will change the target nominal centerpoint on the display

  if (AZ > 600) AZsetpoint -= 10;
  if (AZ < 400) AZsetpoint += 10;
  if (EL > 600) ELsetpoint -= 5;
  if (EL < 400) ELsetpoint += 5;
}

//-------------------------------------------------------------------------------
void CenterJoystick()  // Read joystick positions when joystick is centered
{
  analogReadResolution(12);
  JoystickCenterAZ = analogRead(AZPin);
  JoystickCenterEL = analogRead(ELPin);
  return;
}

//-----------------------------------------------------------------------------------------------------
void AZStepperMode16() {
  digitalWrite(AZModePin1, HIGH);  // 1/16 stepping
  digitalWrite(AZModePin2, HIGH);
  digitalWrite(AZModePin3, HIGH);

  AZStepperMode = 16;  // 1/16 stepper mode
}

//-----------------------------------------------------------------------------------------------------
void ELStepperMode16() {
  digitalWrite(ELModePin1, HIGH);  // 1/16 stepping
  digitalWrite(ELModePin2, HIGH);
  digitalWrite(ELModePin3, HIGH);

  ELStepperMode = 16;  // 1/16 stepper mode
}

void EnableSteppers() {
  digitalWrite(StepperEnablePin, LOW);
  SteppersEnabled = true;
}

void DisableSteppers() {
  digitalWrite(StepperEnablePin, HIGH);
  SteppersEnabled = false;
}

/* ======================================================================
  Test Thrustmaster Joystick Position
  ====================================================================== */
void ReadJoystick()  // A test module to condinuously read and report joystick positions
{
  int JoystickAZPinReading;
  int JoystickELPinReading;
  analogReadResolution(12);
  JoystickAZPinReading = analogRead(AZPin);
  JoystickELPinReading = analogRead(ELPin);
  return;
}

static inline void enablePioClock(uint32_t periphId) {
  if (periphId < 32) {
    PMC->PMC_PCER0 = (1u << periphId);
  } else {
    PMC->PMC_PCER1 = (1u << (periphId - 32));
  }
}

static inline uint8_t fastDigitalRead(uint32_t pin) {
  Pio* p = g_APinDescription[pin].pPort;
  uint32_t m = g_APinDescription[pin].ulPin;
  return (p->PIO_PDSR & m) ? 1 : 0;
}

// =================================================
// Timer Setup (does NOT start the timer)
// =================================================
void setupTimerChannel(uint32_t channel) {
  pmc_set_writeprotect(false);

  // Enable peripheral clock for TC0 channels
  pmc_enable_periph_clk(ID_TC0 + channel);

  // Disable clock
  TC0->TC_CHANNEL[channel].TC_CCR = TC_CCR_CLKDIS;

  // Disable interrupts
  TC0->TC_CHANNEL[channel].TC_IDR = 0xFFFFFFFF;

  // Clear status register
  TC0->TC_CHANNEL[channel].TC_SR;

  // Configure waveform mode, count up to RC
  TC0->TC_CHANNEL[channel].TC_CMR =
    TC_CMR_WAVE | TC_CMR_WAVSEL_UP_RC | TC_CMR_TCCLKS_TIMER_CLOCK3;

  // Enable interrupt on RC compare
  TC0->TC_CHANNEL[channel].TC_IER = TC_IER_CPCS;

  if (channel == 0)
    NVIC_EnableIRQ(TC0_IRQn);
  else if (channel == 1)
    NVIC_EnableIRQ(TC1_IRQn);
}

// =================================================
// Frequency Control
// =================================================
void setTone1Frequency(uint32_t frequency) {
  uint32_t rc = TIMER_CLOCK / (frequency * 2);
  TC0->TC_CHANNEL[0].TC_RC = rc;

  // Reset counter for clean transition
  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_SWTRG;
}

void setTone2Frequency(uint32_t frequency) {
  uint32_t rc = TIMER_CLOCK / (frequency * 2);
  TC0->TC_CHANNEL[1].TC_RC = rc;

  // Reset counter for clean transition
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_SWTRG;
}

// =================================================
// Start / Stop Control
// =================================================
void startTone1() {
  tone1Running = true;
  TC0->TC_CHANNEL[0].TC_CCR =
    TC_CCR_CLKEN | TC_CCR_SWTRG;
}

void stopTone1() {
  tone1Running = false;
  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKDIS;
  digitalWrite(PIN_TONE1, LOW);
}

void startTone2() {
  tone2Running = true;
  TC0->TC_CHANNEL[1].TC_CCR =
    TC_CCR_CLKEN | TC_CCR_SWTRG;
}

void stopTone2() {
  tone2Running = false;
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_CLKDIS;
  digitalWrite(PIN_TONE2, LOW);
}

// =================================================
// Interrupt Handlers
// =================================================
void TC0_Handler() {
  TC0->TC_CHANNEL[0].TC_SR;

  if (tone1Running)
    g_APinDescription[PIN_TONE1].pPort->PIO_ODSR ^=
      g_APinDescription[PIN_TONE1].ulPin;
}

void TC1_Handler() {
  TC0->TC_CHANNEL[1].TC_SR;

  if (tone2Running)
    g_APinDescription[PIN_TONE2].pPort->PIO_ODSR ^=
      g_APinDescription[PIN_TONE2].ulPin;
}

// Universal direct-register digital pin read.  Returns value HIGH or LOW
int fastPinState(uint8_t pin) {
  Pio* port = g_APinDescription[pin].pPort;
  uint32_t mask = g_APinDescription[pin].ulPin;

  return (port->PIO_PDSR & mask) ? HIGH : LOW;
}
