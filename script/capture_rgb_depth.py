#!/usr/bin/env python3
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##       Align Depth to Color (Dynamic Res)        ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os
import argparse
from datetime import datetime

def main():
    # ---------------------- #
    #  1. Parse Arguments    #
    # ---------------------- #
    parser = argparse.ArgumentParser(description="RealSense alignment script with dynamic resolution.")
    parser.add_argument("--width", type=int, default=640, help="Color/Depth width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Color/Depth height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--clipping_distance", type=float, default=1.0,
                        help="Clipping distance in meters (default: 1.0m)")
    args = parser.parse_args()

    width = args.width
    height = args.height
    fps = args.fps
    clipping_distance_in_meters = args.clipping_distance

    # ---------------------- #
    #  2. Start Pipeline     #
    # ---------------------- #
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    device = profile.get_device()

    # Check for RGB sensor
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The script requires a Depth camera with a Color sensor.")
        pipeline.stop()
        return

    # Get the depth sensor's scale
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Calculate clipping distance
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object (align depth to color)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ------------------------------ #
    #  3. Prepare Output Directories #
    # ------------------------------ #
    base_dir = "/home/martyn/Thesis/realsense-scripts/"
    res_folder = f"{width}x{height}"
    out_base = os.path.join(base_dir, "out", res_folder)
    os.makedirs(out_base, exist_ok=True)

    # ----------------------------- #
    #  4. Save Intrinsics Once     #
    # ----------------------------- #
    # We'll do one wait_for_frames() to ensure we can retrieve intrinsics properly.
    frames_test = pipeline.wait_for_frames()
    aligned_test = align.process(frames_test)
    aligned_depth_test = aligned_test.get_depth_frame()

    if aligned_depth_test:
        # Retrieve intrinsics of the aligned depth frame
        intrinsics = aligned_depth_test.profile.as_video_stream_profile().intrinsics
        cam_K = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]

        # Paths to intrinsics files in out/<width>x<height>
        camK_txt = os.path.join(out_base, "cam_K.txt")
        camK_json = os.path.join(out_base, "cam_K.json")

        # Only save if not already existing (one time per resolution)
        if not os.path.exists(camK_txt) or not os.path.exists(camK_json):
            # Save intrinsics .txt
            with open(camK_txt, "w") as f:
                f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                f.write(f"{0.0} {0.0} {1.0}\n")

            # Save intrinsics .json
            camera_intrinsics = {
                "cam_K": cam_K,
                "depth_scale": depth_scale
            }
            with open(camK_json, "w") as f:
                json.dump(camera_intrinsics, f, indent=4)

            print(f"Intrinsic parameters saved to:\n  {camK_txt}\n  {camK_json}")
        else:
            print(f"Intrinsics for this resolution already exist in:\n  {camK_txt}\n  {camK_json}")
    else:
        print("Unable to retrieve depth frame for intrinsics. Please check camera configuration.")

    # ----------------------- #
    #  5. Main Loop + Logic   #
    # ----------------------- #
    RecordStream = False
    recording_folder = None
    start_time = 0
    elapsed_time = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            unaligned_depth_frame = frames.get_depth_frame()
            unaligned_color_frame = frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
            unaligned_rgb_image = np.asanyarray(unaligned_color_frame.get_data())

            # Remove background (beyond clipping distance)
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where(
                (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                grey_color,
                color_image
            )

            # Create colorized depth image for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))

            # If recording, show elapsed time
            if RecordStream:
                elapsed_time = time.time() - start_time
                cv2.putText(
                    images, f"Recording: {elapsed_time:.2f} s",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA
                )

            # Show images
            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)

            key = cv2.waitKey(1)
            if key < 0:
                pass

            # ---------- Toggle Recording (space) ----------
            if key & 0xFF == ord(" "):
                if not RecordStream:
                    # Start recording
                    time.sleep(0.2)
                    RecordStream = True
                    start_time = time.time()

                    # Create a unique folder for this particular recording
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_folder = os.path.join(out_base, timestamp_str)
                    os.makedirs(recording_folder, exist_ok=True)

                    for subfolder in ["depth", "rgb", "depth_unaligned", "rgb_unaligned"]:
                        os.makedirs(os.path.join(recording_folder, subfolder), exist_ok=True)

                    print(f"Recording started. Saving frames to:\n  {recording_folder}")
                else:
                    # Stop recording
                    RecordStream = False
                    elapsed_time = time.time() - start_time
                    print(f"Recording stopped. Duration: {elapsed_time:.2f} seconds")

            # ---------- Save frames if recording ----------
            if RecordStream and recording_folder:
                framename = int(round(time.time() * 1000))

                image_path_depth = os.path.join(recording_folder, "depth", f"{framename}.png")
                image_path_rgb = os.path.join(recording_folder, "rgb", f"{framename}.png")
                image_path_depth_unaligned = os.path.join(recording_folder, "depth_unaligned", f"{framename}.png")
                image_path_rgb_unaligned = os.path.join(recording_folder, "rgb_unaligned", f"{framename}.png")

                cv2.imwrite(image_path_depth, depth_image)
                cv2.imwrite(image_path_rgb, color_image)
                cv2.imwrite(image_path_depth_unaligned, unaligned_depth_image)
                cv2.imwrite(image_path_rgb_unaligned, unaligned_rgb_image)

            # ---------- Single frame capture (c) ----------
            if key & 0xFF == ord("c"):
                single_capture_folder = os.path.join(out_base, "out_single")
                os.makedirs(single_capture_folder, exist_ok=True)

                subfolder_depth_foundationpose = os.path.join(single_capture_folder, "depth")
                subfolder_rgb_foundationpose = os.path.join(single_capture_folder, "rgb")
                subfolder_depth_unaligned_foundationpose = os.path.join(single_capture_folder, "depth_unaligned")
                subfolder_rgb_unaligned_foundationpose = os.path.join(single_capture_folder, "rgb_unaligned")

                for folder in [
                    subfolder_depth_foundationpose,
                    subfolder_rgb_foundationpose,
                    subfolder_depth_unaligned_foundationpose,
                    subfolder_rgb_unaligned_foundationpose
                ]:
                    os.makedirs(folder, exist_ok=True)

                framename = int(round(time.time() * 1000))
                image_path_depth_foundationpose = os.path.join(subfolder_depth_foundationpose, f"{framename}.png")
                image_path_rgb_foundationpose = os.path.join(subfolder_rgb_foundationpose, f"{framename}.png")
                image_path_depth_unaligned_foundationpose = os.path.join(subfolder_depth_unaligned_foundationpose, f"{framename}.png")
                image_path_rgb_unaligned_foundationpose = os.path.join(subfolder_rgb_unaligned_foundationpose, f"{framename}.png")

                cv2.imwrite(image_path_depth_foundationpose, depth_image)
                cv2.imwrite(image_path_rgb_foundationpose, color_image)
                cv2.imwrite(image_path_depth_unaligned_foundationpose, unaligned_depth_image)
                cv2.imwrite(image_path_rgb_unaligned_foundationpose, unaligned_rgb_image)

                print(f"Captured single frame and saved to 'out_single' at {framename}.png")

            # ---------- Quit on 'q' or Esc ----------
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
