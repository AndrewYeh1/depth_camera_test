import cv2
import numpy as np

def remove_fisheye_and_warp_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return

    # Grab the true width and height of the video source
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Because you added cv2.rotate(..., ROTATE_90_CLOCKWISE), the image is now taller than it is wide.
    # We MUST swap w and h so the matrices are created for the newly rotated dimensions!
    w, h = real_h, real_w

    # ==========================================
    # 1. SETUP FISHEYE UNDISTORTION
    # ==========================================
    # Replace these dummy parameters with your actual calibration parameters.
    camera_matrix = np.array([[w, 0, w/2],
                              [0, h, h/2],
                              [0, 0, 1]], dtype=np.float32)
                              
    # Negative k1 typically undoes barrel (fisheye) distortion.
    dist_coeffs = np.array([-0.85, 0.0, 0.0, 0.0, 0.0]) 

    # ==========================================
    # 2. SETUP PERSPECTIVE WARP (Bird's Eye View)
    # ==========================================
    # Define 4 source points on the undistorted frame.
    # To "pan the camera to the right", we shift all these X-coordinates further to the right.
    # This grabs a slice of the original image that's further right, mimicking a rightward pan.
    src_points = np.float32([
        [0, 0],            # Top-left
        [w, h * 0.3],            # Top-right
        [0, h],            # Bottom-left
        [w, h * 0.75]             # Bottom-right
    ])

    # Define the 4 destination points you want them to map to
    dst_points = np.float32([
        [0, 0],            # Top-left
        [w, 0],            # Top-right
        [0, h],            # Bottom-left
        [w, h]             # Bottom-right
    ])

    # Calculate the perspective transform matrix ONCE outside the loop since it doesn't change
    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    print("Playing video... Press 'q' to quit.")
    display_scale = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
            
        # Rotate the frame first so ALL calculations (undistort, warp) use the correct tall orientation!
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # ------------------------------------------
        # 1. APPLY UNDISTORTION
        # ------------------------------------------
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # ------------------------------------------
        # 2. APPLY PERSPECTIVE WARP
        # ------------------------------------------
        warped_frame = cv2.warpPerspective(undistorted_frame, homography_matrix, (w, h))

        # ------------------------------------------
        # 3. STRETCH TO HORIZONTAL 16:9
        # ------------------------------------------
        # Since the video was rotated to be tall (w=1080, h=1920), 
        # stretching it to horizontal 16:9 gives a standard 1920x1080 output.
        target_w = int(w * 1.8)
        warped_frame = cv2.resize(warped_frame, (target_w, w))

        # ------------------------------------------
        # 4. HOW TO CROP THE IMAGE
        # ------------------------------------------
        # In OpenCV, images are just NumPy arrays. Cropping is simply array slicing!
        # Syntax: image[y_start:y_end, x_start:x_end]
        
        # Example: Crop 50 pixels off the top and bottom, and 100 off left and right.
        h_warped, w_warped = warped_frame.shape[:2]
        crop_top = 50
        crop_bottom = h_warped - 300
        crop_left = 0
        crop_right = w_warped
        
        warped_frame = warped_frame[crop_top:crop_bottom, crop_left:crop_right]

        # ------------------------------------------
        # DISPLAY RESULTS
        # ------------------------------------------
        # Draw circles on the point sources to visualize the perspective crop area
        viz_frame = undistorted_frame.copy()
        for pt in src_points:
            cv2.circle(viz_frame, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)

        cv2.imshow("1. Original Video", cv2.resize(frame, (0,0), fx=display_scale, fy=display_scale))
        cv2.imshow("2. Undistorted Lens", cv2.resize(viz_frame, (0,0), fx=display_scale, fy=display_scale))
        cv2.imshow("3. Perspective Warped", cv2.resize(warped_frame, (0,0), fx=display_scale, fy=display_scale))
        
        # Press 'q' to exit. waitKey(1) makes the video play continuously without pausing.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with the video file used in your other script
    remove_fisheye_and_warp_video('WIN_20260326_10_11_43_Pro.mp4')
