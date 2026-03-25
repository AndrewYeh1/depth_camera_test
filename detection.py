import cv2
import numpy as np
import torch
from ultralytics import YOLO

def main():
    # Detect if CUDA is available (crucial for NVIDIA Jetson)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running inference on device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    # Load YOLO11 instance segmentation model
    # 'yolo11n-seg.pt' is the newest lightweight layout for real-time inference
    print("Loading YOLO model...")
    model = YOLO("yolo11n-seg.pt")

    cap = cv2.VideoCapture(0)

    # In the depth camera project, 2560x960 implies side-by-side feeds
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Since it's a stereo camera, split horizontal width exactly in half
        height, width, _ = frame.shape
        half_width = width // 2
        
        # We will use the left camera feed to process the detection
        left_feed = frame[:, :half_width]

        # Class index 1 represents 'bicycle' in the COCO dataset.
        # Jetson Nano Optimizations: 
        #   imgsz=416 -> drastically drops tensor footprint.
        #   half=True (FP16) -> Jetson architectures excel at half-precision math.
        #   device=x  -> enforce running purely statically on GPU architecture.
        results = model.predict(
            source=left_feed, 
            classes=[1], 
            conf=0.3, 
            verbose=False,
            imgsz=416, 
            half=(device == 0), 
            device=device
        )

        # Make a clone to draw outlines and masks onto without traditional bounding boxes
        annotated_frame = left_feed.copy()

        # Check results and render masks if there's any valid objects mapping to our requested class
        for result in results:
            if result.masks is not None:
                # `result.masks.xy` holds individual polygon contours scaled securely to our original dimension
                segments = result.masks.xy 

                for segment in segments:
                    # Parse valid x,y polygons as standard [int32] numpy arrays for OpenCV
                    segment_pts = np.array(segment, dtype=np.int32)
                    
                    # 1. Outline: Draw a solid green outline wrapping gracefully around the bicycle
                    cv2.polylines(annotated_frame, [segment_pts], isClosed=True, color=(0, 255, 0), thickness=3)

                    # 2. Mask: Draw an explicit transparent overlay filtering only the bounds of the bicycle
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [segment_pts], color=(0, 255, 0))
                    
                    # Alpha-blend the polygon filling over the original structure (0.3 translates to a ~30% tint)
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Display output
        cv2.imshow("Bicycle Mask Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release tracking and rendering pipeline
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
