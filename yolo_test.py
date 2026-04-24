import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model (downloads 'yolov8n.pt' if not present locally)
    model = YOLO('yolo26m.pt')

    # Open the video file (change the path if needed)
    video_path = 'videos/2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # COCO class IDs: 0 is 'person', 1 is 'bicycle'
    # By passing these classes to the model, it will only return predictions for them.
    target_classes = [0, 1]

    print("Press 'q' to quit the video playback.")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame, filtering for our target classes
            # device='cuda:0' forces the use of the primary GPU
            results = model(frame, classes=target_classes, verbose=False, device='cuda:0')

            # Draw the bounding boxes and labels on the frame
            annotated_frame = results[0].plot()

            # Resize the display frame so it fits on most screens
            # You can adjust or remove this if you want the original resolution
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("YOLO - Bikes and People Detection", display_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Reached the end of the video
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
