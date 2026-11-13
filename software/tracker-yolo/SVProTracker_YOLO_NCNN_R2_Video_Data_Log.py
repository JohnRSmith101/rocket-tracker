#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 John R. Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# This program uses Ultralytics YOLO, which is licensed under AGPL-3.0.
# Based on Ultralytics YOLO11n model deployed in NCNN format.
# See the NOTICE file.

###############################################################
# Raspberry Pi 5 rocket detection and Arduino gimbal tracking
# Ultralytics YOLO11n model deployed in NCNN format
#
# Selects only the highest-confidence detection in each frame.
# All eight trained classes are eligible for tracking:
# staging, blob, plume, backlit_plume, jellyfish, contrail,
# engines, point
#
# Current Arduino packet:
# raw X, raw Y, raw azimuth velocity, raw elevation velocity
#
# Predicted position and filtered/tapered feedforward values are
# still calculated and retained for possible future use.
###############################################################

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pygame
import pygame.camera
import serial
from pygame import mixer
from ultralytics import YOLO


###############################################################
# User-editable configuration
###############################################################

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1.0

REQUESTED_IMG_W = 960
REQUESTED_IMG_H = 540

DEFAULT_CONFIDENCE_THRESHOLD = 0.30
YOLO_IMAGE_SIZE = 640

RECORDING_FPS = 30.0
RECORDING_CODEC = "mp4v"
ALERT_SOUND_FILE = "alert.wav"

EXPECTED_CLASSES = {
    "staging",
    "blob",
    "plume",
    "backlit_plume",
    "jellyfish",
    "contrail",
    "engines",
    "point",
}

FOV_X_DEG = 5.20 # was 3.20
FOV_Y_DEG = 4.01 # was 1.81

TAU_MIN = 0.05
TAU_MAX = 1.0
MEASURED_LAG = 0.413

# Larger values give more weight to the previous filtered value.
VELOCITY_FILTER_ALPHA = 0.90

# A value of 1.0 uses the current measured position with no smoothing.
POSITION_ALPHA = 1.0

FF_TAPER_AZ_PX = 40
FF_TAPER_EL_PX = 40
CENTER_HOLD_AZ_PX = 3
CENTER_HOLD_EL_PX = 3
VEL_DEADBAND_DEG_S = 0.002

FF_FULL_FPS = 7.0
FF_REDUCED_FPS = 4.0
FF_MIN_SCALE = 0.2
LOW_FPS_DISABLE_FF = 3.0
TAU_LOW_FPS_SCALE = 0.85

MIN_SEND_INTERVAL = 0.05


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO NCNN rocket detector and Arduino gimbal tracker"
    )
    parser.add_argument(
        "--modeldir",
        required=True,
        help=(
            "Path to the exported NCNN model directory, for example "
            "YOLOmodel/my_model_ncnn_model"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "Minimum detection confidence. Default is "
            f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}"
        ),
    )
    return parser.parse_args()


def validate_model_classes(model):
    """Print a warning if the model's class names differ from expectations."""
    names = model.names
    if isinstance(names, dict):
        actual_names = {str(name) for name in names.values()}
    else:
        actual_names = {str(name) for name in names}

    if actual_names == EXPECTED_CLASSES:
        print("Model classes verified:", ", ".join(sorted(actual_names)))
        return

    print("WARNING: Model class names differ from the expected eight classes.")
    print("Expected:", ", ".join(sorted(EXPECTED_CLASSES)))
    print("Model contains:", ", ".join(sorted(actual_names)))


def start_recording(frame_width, frame_height):
    """Create synchronized MP4 and CSV recording files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = f"recording_{timestamp}.mp4"
    log_filename = f"recording_{timestamp}.csv"

    fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
    writer = cv2.VideoWriter(
        video_filename,
        fourcc,
        RECORDING_FPS,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not open video file for writing: {video_filename}"
        )

    try:
        log_file = open(log_filename, "w", newline="", buffering=8192)
    except Exception:
        writer.release()
        raise

    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            "Frame",
            "Time_sec",
            "Target_Detected",
            "Target_X",
            "Target_Y",
            "AZ_Vel_deg_s",
            "EL_Vel_deg_s",
            "Filtered_AZ_Vel_deg_s",
            "Filtered_EL_Vel_deg_s",           
            "AZ_Error_px",
            "EL_Error_px",
            "Serial_Sent",
        ]
    )

    print(f"### Recording started: {video_filename}")
    print(f"### Data logging started: {log_filename}")
    return writer, log_file, log_writer, time.perf_counter()


def stop_recording(video_writer, log_file):
    """Close the active MP4 and CSV recording files."""
    if video_writer is not None:
        video_writer.release()
    if log_file is not None:
        log_file.close()
    print("### Recording and data logging stopped")


def get_model_class_name(model, class_id):
    """Return a class name for either dict-style or list-style model names."""
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def main():
    args = parse_arguments()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")

    model_path = os.path.abspath(args.modeldir)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"NCNN model directory not found: {model_path}")

    print(f"Loading NCNN model from: {model_path}")
    model = YOLO(model_path)
    validate_model_classes(model)

    serial_connection = None
    camera = None
    video_writer = None
    log_file = None
    log_writer = None
    recording_start_time = None
    recording_frame_number = 0
    recording = False
    recording_state = "Not Recording"

    try:
        serial_connection = serial.Serial(
            SERIAL_PORT,
            SERIAL_BAUD,
            timeout=SERIAL_TIMEOUT,
        )
        time.sleep(0.01)
        serial_connection.reset_input_buffer()
        arduino_message = " "

        pygame.init()
        pygame.camera.init()
        mixer.init()
        alert = mixer.Sound(ALERT_SOUND_FILE)

        cameras = pygame.camera.list_cameras()
        if not cameras:
            raise RuntimeError("No cameras found.")

        camera_id = cameras[0]
        print(f"Using camera: {camera_id}")

        camera = pygame.camera.Camera(
            camera_id,
            (REQUESTED_IMG_W, REQUESTED_IMG_H),
        )
        camera.start()
        time.sleep(2.0)

        last_send = 0.0
        previous_smoothed_x = REQUESTED_IMG_W / 2.0
        previous_smoothed_y = REQUESTED_IMG_H / 2.0

        previous_measurement_x = None
        previous_measurement_y = None
        previous_measurement_time = None

        filtered_vx_deg_s = 0.0
        filtered_vy_deg_s = 0.0

        tick_frequency = cv2.getTickFrequency()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            if serial_connection.in_waiting > 0:
                arduino_message = (
                    serial_connection.readline()
                    .decode("utf-8", errors="replace")
                    .strip()
                )

            time.sleep(0.001)
            frame_start_tick = cv2.getTickCount()

            pygame_surface = camera.get_image()
            pygame_rgb = pygame.surfarray.array3d(pygame_surface)
            frame = cv2.cvtColor(
                pygame_rgb.transpose(1, 0, 2),
                cv2.COLOR_RGB2BGR,
            )

            # This lookup is negligible compared with YOLO inference.
            img_h, img_w = frame.shape[:2]
            center_x = img_w // 2
            center_y = img_h // 2

            # Per-frame values used by the synchronized CSV logger.
            target_detected = False
            log_target_x = None
            log_target_y = None
            log_vx_deg_s = None
            log_vy_deg_s = None
            log_filtered_vx_deg_s = None
            log_filtered_vy_deg_s = None
            log_az_error_px = None
            log_el_error_px = None
            serial_sent = False

            results = model.predict(
                source=frame,
                conf=args.threshold,
                imgsz=YOLO_IMAGE_SIZE,
                verbose=False,
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                best_index = int(np.argmax(confidences))
                best_box = result.boxes[best_index]

                confidence = float(best_box.conf.item())
                class_id = int(best_box.cls.item())
                xmin, ymin, xmax, ymax = (
                    best_box.xyxy[0].cpu().numpy().round().astype(int)
                )

                xmin = max(0, min(img_w - 1, xmin))
                xmax = max(0, min(img_w - 1, xmax))
                ymin = max(0, min(img_h - 1, ymin))
                ymax = max(0, min(img_h - 1, ymax))

                target_x = (xmin + xmax) // 2
                target_y = (ymin + ymax) // 2

                target_x_raw = max(0, min(img_w - 1, target_x))
                target_y_raw = max(0, min(img_h - 1, target_y))
                target_detected = True

                target_x_smoothed = (
                    POSITION_ALPHA * target_x_raw
                    + (1.0 - POSITION_ALPHA) * previous_smoothed_x
                )
                target_y_smoothed = (
                    POSITION_ALPHA * target_y_raw
                    + (1.0 - POSITION_ALPHA) * previous_smoothed_y
                )
                previous_smoothed_x = target_x_smoothed
                previous_smoothed_y = target_y_smoothed

                now = time.time()
                tau = max(TAU_MIN, min(TAU_MAX, MEASURED_LAG))

                if previous_measurement_time is None:
                    measurement_interval = MIN_SEND_INTERVAL
                    fps_estimate = 0.0
                    vx_px_s = 0.0
                    vy_px_s = 0.0
                else:
                    measurement_interval = now - previous_measurement_time
                    if measurement_interval < 1e-3:
                        measurement_interval = MIN_SEND_INTERVAL

                    fps_estimate = 1.0 / measurement_interval
                    vx_px_s = (
                        target_x_raw - previous_measurement_x
                    ) / measurement_interval
                    vy_px_s = (
                        target_y_raw - previous_measurement_y
                    ) / measurement_interval

                degrees_per_pixel_x = FOV_X_DEG / float(img_w)
                degrees_per_pixel_y = FOV_Y_DEG / float(img_h)
                vx_deg_s = vx_px_s * degrees_per_pixel_x
                vy_deg_s = vy_px_s * degrees_per_pixel_y
                filtered_vx_deg_s = (
                    VELOCITY_FILTER_ALPHA * filtered_vx_deg_s
                    + (1.0 - VELOCITY_FILTER_ALPHA) * vx_deg_s
                )
                filtered_vy_deg_s = (
                    VELOCITY_FILTER_ALPHA * filtered_vy_deg_s
                    + (1.0 - VELOCITY_FILTER_ALPHA) * vy_deg_s
                )

                # These are the raw values used in the current Arduino packet.
                # Pixel errors are signed: positive/negative preserve direction.
                log_target_x = target_x_raw
                log_target_y = target_y_raw
                log_vx_deg_s = vx_deg_s
                log_vy_deg_s = vy_deg_s
                log_filtered_vx_deg_s = filtered_vx_deg_s
                log_filtered_vy_deg_s = filtered_vy_deg_s
                log_az_error_px = target_x_raw - center_x
                log_el_error_px = target_y_raw - center_y



                if fps_estimate >= FF_FULL_FPS:
                    feedforward_rate_scale = 1.0
                    tau_to_use = tau
                elif fps_estimate >= FF_REDUCED_FPS:
                    feedforward_rate_scale = 0.5
                    tau_to_use = tau
                elif fps_estimate >= LOW_FPS_DISABLE_FF:
                    feedforward_rate_scale = FF_MIN_SCALE
                    tau_to_use = tau * TAU_LOW_FPS_SCALE
                else:
                    feedforward_rate_scale = 0.0
                    tau_to_use = tau * TAU_LOW_FPS_SCALE

                predicted_x = target_x_smoothed + vx_px_s * tau_to_use
                predicted_y = target_y_smoothed + vy_px_s * tau_to_use
                predicted_x = max(0.0, min(img_w - 1.0, predicted_x))
                predicted_y = max(0.0, min(img_h - 1.0, predicted_y))

                azimuth_error_px = abs(target_x_smoothed - center_x)
                elevation_error_px = abs(target_y_smoothed - center_y)

                feedforward_scale_az = min(
                    1.0,
                    azimuth_error_px / float(FF_TAPER_AZ_PX),
                )
                feedforward_scale_el = min(
                    1.0,
                    elevation_error_px / float(FF_TAPER_EL_PX),
                )

                vx_to_send_later = (
                    filtered_vx_deg_s
                    * feedforward_scale_az
                    * feedforward_rate_scale
                )
                vy_to_send_later = (
                    filtered_vy_deg_s
                    * feedforward_scale_el
                    * feedforward_rate_scale
                )

                if abs(vx_to_send_later) < VEL_DEADBAND_DEG_S:
                    vx_to_send_later = 0.0
                if abs(vy_to_send_later) < VEL_DEADBAND_DEG_S:
                    vy_to_send_later = 0.0

                if azimuth_error_px < CENTER_HOLD_AZ_PX:
                    predicted_x = float(center_x)
                    vx_to_send_later = 0.0
                if elevation_error_px < CENTER_HOLD_EL_PX:
                    predicted_y = float(center_y)
                    vy_to_send_later = 0.0

                # Current packet
                if now - last_send >= MIN_SEND_INTERVAL:
                    last_send = now
                    packet = (
                        f"{target_x_raw:.2f},"
                        f"{target_y_raw:.2f},"
                        f"{filtered_vx_deg_s:.6f},"
                        f"{filtered_vy_deg_s:.6f}\n"
                    ).encode("utf-8")
                    serial_connection.write(packet)
                    serial_sent = True

                previous_measurement_x = target_x_raw
                previous_measurement_y = target_y_raw
                previous_measurement_time = now

                displayed_target_x = int(round(target_x_smoothed))
                displayed_target_y = int(round(target_y_smoothed))

                alert.play()

                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    (10, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Target: {displayed_target_x},{displayed_target_y}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                object_name = get_model_class_name(model, class_id)
                label = f"{object_name}: {int(confidence * 100)}%"
                label_y = max(25, ymin - 10)
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Retained for future packet testing:
                # predicted_x, predicted_y,
                # vx_to_send_later, vy_to_send_later.

            frame_end_tick = cv2.getTickCount()
            elapsed_time = (
                frame_end_tick - frame_start_tick
            ) / tick_frequency
            frame_rate = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

            cv2.putText(
                frame,
                arduino_message,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "+",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {frame_rate:.2f}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                recording_state,
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if recording and video_writer is not None:
                video_writer.write(frame)
                recording_frame_number += 1

                if log_writer is not None and recording_start_time is not None:
                    log_writer.writerow(
                        [
                            recording_frame_number,
                            f"{time.perf_counter() - recording_start_time:.6f}",
                            1 if target_detected else 0,
                            log_target_x if target_detected else "",
                            log_target_y if target_detected else "",
                            (
                                f"{log_vx_deg_s:.6f}"
                                if target_detected
                                else ""
                            ),
                            (
                                f"{log_vy_deg_s:.6f}"
                                if target_detected
                                else ""
                            ),
                              (
                                f"{log_filtered_vx_deg_s:.6f}"
                                if target_detected
                                else ""
                            ),
                            (
                                f"{log_filtered_vy_deg_s:.6f}"
                                if target_detected
                                else ""
                            ),
                            log_az_error_px if target_detected else "",
                            log_el_error_px if target_detected else "",
                            1 if serial_sent else 0,
                        ]
                    )

            cv2.imshow("Object Detector", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                running = False

            elif key == ord("r"):
                if not recording:
                    (
                        video_writer,
                        log_file,
                        log_writer,
                        recording_start_time,
                    ) = start_recording(img_w, img_h)
                    recording_frame_number = 0
                    recording = True
                    recording_state = "Recording"
                else:
                    stop_recording(video_writer, log_file)
                    video_writer = None
                    log_file = None
                    log_writer = None
                    recording_start_time = None
                    recording = False
                    recording_state = "Not Recording"

    finally:
        if video_writer is not None:
            video_writer.release()
        if log_file is not None:
            log_file.close()

        if camera is not None:
            camera.stop()

        if serial_connection is not None and serial_connection.is_open:
            serial_connection.close()

        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
