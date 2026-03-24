import cv2

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

        # Note: Depending on the specific physical setup of your camera lenses, 
        # left could be visual and right could be depth/IR, or vice versa.
        # Swap these assignments if your setup is reversed!
        visual_feed = left_feed
        depth_feed = right_feed

        # You might want to process the 'depth_feed' depending on how it's returned.
        # If it's an IR camera for depth mapping, it might just look like a grayscale image.

        # Display the separate feeds
        cv2.imshow('Visual Feed', visual_feed)
        cv2.imshow('Depth Feed', depth_feed)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
