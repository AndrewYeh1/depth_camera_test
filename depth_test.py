import cv2
import numpy as np

def main():
    # Open the default camera (index 0). You might need to change this index
    # if you have multiple cameras connected (e.g., to 1, 2, etc.)
    cap = cv2.VideoCapture(0)

    # For the ELP-USB960p2CAM-V90, the combined resolution is often 2560x960.
    # Set the resolution to the maximum combined size to ensure you get both feeds side-by-side.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    # Some cameras perform better if we explicitly set the MJPEG codec to handle high resolutions
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # The image from this stereo camera is a side-by-side merged frame.
        # We split the frame in half horizontally to get the two individual feeds.
        height, width, _ = frame.shape
        half_width = width // 2
        
        # Split into left and right frames
        left_feed = frame[:, :half_width]
        right_feed = frame[:, half_width:]

        # OPTIMIZATION 1: Downsample the color images to speed up computation
        # (e.g., resizing to 50% width and height)
        scale = 0.5
        left_small = cv2.resize(left_feed, (0, 0), fx=scale, fy=scale)
        right_small = cv2.resize(right_feed, (0, 0), fx=scale, fy=scale)

        # OPTIMIZATION 2: Tweak StereoSGBM Parameters for Speed
        min_disp = 0
        num_disp = 16 * 3  # Reduced from 80 to 48 (must be approx divisible by 16)
        block_size = 5
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,  # Reduced for speed
            speckleRange=16,       # Reduced for speed
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # OPTIMIZATION 3: Use 3-way SGBM (faster)
        )

        # Compute Disparity on the smaller color images
        disparity_small = stereo.compute(left_small, right_small).astype(np.float32) / 16.0
        
        # OPTIMIZATION 4: Resize disparity back to original size for display
        disparity = cv2.resize(disparity_small, (width // 2, height), interpolation=cv2.INTER_LINEAR)
        # Disparity values scale linearly with image resolution, so multiply by (1/scale) to restore true magnitude
        disparity = disparity / scale
        
        # Avoid division by zero and invalid disparity values (-1.0 or <= 0)
        disparity[disparity <= 0] = 0.1

        # Camera Parameters (Estimated)
        focal_length = 640.0  # Pixels, based on 1280px width per eye and ~90 degree horizontal FOV
        baseline = 60.0       # Millimeters, typical for this type of binocular camera
        
        # Calculate Depth map (Z = (f * B) / d). Units will be in millimeters.
        depth = (focal_length * baseline) / disparity
        
        # Normalize disparity map for display
        # The normalization is mapped linearly to 0-255 purely for visualizing it on screen.
        disp_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply pseudo-color map for better visualization (Red=Close, Blue=Far)
        # Note: Since higher disparity means closer objects, the pseudo-color will highlight them natively.
        color_depth = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)

        # Display the feeds
        cv2.imshow('Left Feed (Visual)', left_feed)
        cv2.imshow('Depth Map (Disparity)', color_depth)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
