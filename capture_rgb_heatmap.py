import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB and depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Create output folders
output_dir = "./captures"
rgb_dir = os.path.join(output_dir, "rgb")
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

try:
    print("Press 'Spacebar' to capture images. Press 'q' to quit.")
    while True:
        # Wait for a frame and align depth to color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert depth and color images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize depth values
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply colormap for heat map
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_normalized), cv2.COLORMAP_JET
        )

        # Stack RGB and depth map for visualization
        stacked_images = np.hstack((color_image, depth_colormap))

        # Show the visualization
        cv2.imshow("RGB and Depth Map", stacked_images)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord(" "):  # Spacebar to capture images
            # Save RGB and depth images
            timestamp = int(round(time.time() * 1000))
            rgb_path = os.path.join(rgb_dir, f"{timestamp}_rgb.png")
            depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")

            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, depth_colormap)

            print(f"Captured RGB: {rgb_path}")
            print(f"Captured Depth Map: {depth_path}")

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
