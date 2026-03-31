import cv2
from ultralytics import YOLO
import torch
import numpy as np
import time

# --- Initialize YOLO ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on device: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
model = YOLO("yolo26l-seg.pt")
model.to(device)

cap = cv2.VideoCapture("WIN_20260326_10_11_43_Pro.mp4")
if not cap.isOpened():
    print("Error opening video stream.")
    exit()

# --- Pre-Calculate Perspective Matrices ---
real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# We rotate 90 degrees clockwise, so swap dimensions for math
w, h = real_h, real_w

camera_matrix = np.array([[w, 0, w/2],
                          [0, h, h/2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.85, 0.0, 0.0, 0.0, 0.0]) 

src_points = np.float32([
    [0, 0],                  # Top-left
    [w, h * 0.3],            # Top-right
    [0, h],                  # Bottom-left
    [w, h * 0.75]            # Bottom-right
])
dst_points = np.float32([
    [0, 0],                  # Top-left
    [w, 0],                  # Top-right
    [0, h],                  # Bottom-left
    [w, h]                   # Bottom-right
])
homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

target_w = int(w)

def apply_perspective_pipeline(frame):
    """Takes a RAW horizontal frame directly from cap.read() and runs the entire warp pipeline"""
    if frame is None:
        return None
    
    # 1. Rotate to correct orientation
    # Only rotate if the frame matches the raw horizontal dimensions
    if frame.shape[:2] == (real_h, real_w):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # 2. Undistort Fisheye
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # 3. Perspective Warp
    warped = cv2.warpPerspective(undistorted, homography_matrix, (w, h))

    # 4. Stretch horizontally
    warped = cv2.resize(warped, (target_w, w))

    # 5. Crop
    h_warped, w_warped = warped.shape[:2]
    crop_top = 50
    crop_bottom = h_warped - 300
    crop_left = 0
    crop_right = w_warped
    
    # Safely crop
    if crop_bottom > crop_top and crop_right > crop_left:
        warped = warped[crop_top:crop_bottom, crop_left:crop_right]
        
    return warped


# --- Load and preprocess reference frame ---
reference = cv2.imread('reference.png')
if reference is not None:
    print("Preprocessing reference.png through warp pipeline...")
    warped_reference = apply_perspective_pipeline(reference)
    ref_blur = cv2.GaussianBlur(warped_reference, (21, 21), 0)
    print("Reference preprocessing complete!")
else:
    print("Warning: reference.png not found. Foreign object detection disabled.")
    ref_blur = None

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame_idx = 0
paused = False

print("\n--- Navigation Controls ---")
print("  Space / 'p' : Play / Pause video")
print("  d           : Skip forward 1 second (or 1 frame if paused)")
print("  a           : Rewind 1 second (or 1 frame if paused)")
print("  q           : Quit")
print("---------------------------\n")

while True:
    loop_start_time = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame_raw = cap.read()
    if not ret:
        print("End of video reached.")
        break
        
    # --- Apply Perspective Warp Pipeline ---
    frame = apply_perspective_pipeline(frame_raw)
        
    # Run YOLO inference natively on the warped frame
    results = model(frame, conf=0.1)
    
    bike_polygons = []
    other_polygons = []
    if results[0].masks is not None:
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 1: # Bicycle / Motorcycle
                bike_polygons.append(np.array(results[0].masks.xy[i], np.int32))
            else: # Other objects YOLO detects
                other_polygons.append(np.array(results[0].masks.xy[i], np.int32))

    ignore_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for pts in bike_polygons:
        cv2.fillPoly(ignore_mask, [pts], 255) # Ignore bike
    for pts in other_polygons:
        cv2.fillPoly(ignore_mask, [pts], 0)   # Do NOT ignore any other object

    # --- Foreign Object Detection via absdiff ---
    if ref_blur is not None:
        frame_blur = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Ensure sizes exactly match before absdiff in case of weird reference.png dimensions
        if ref_blur.shape == frame_blur.shape:
            diff_color = cv2.absdiff(ref_blur, frame_blur)
            diff_intensity = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
            
            # Mask out the bike from the background difference
            diff_intensity[ignore_mask == 255] = 0
                
            _, thresh = cv2.threshold(diff_intensity, 50, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=4)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 2000:
                    continue
                (x, y, box_w, box_h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 0, 255), 2)
                cv2.putText(frame, "Foreign Obj", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print("ERROR: reference.png warped dimensions do not match live camera! Ignoring background sub.")
            ref_blur = None # Disable to prevent terminal spam
            
    # Black out the bicycle in the display window
    frame[ignore_mask == 255] = (0, 0, 0)

    # Draw blue bounding boxes around all objects YOLO detected
    if results[0].boxes is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Scale the HUGE output frame down a bit so it fits on screen
    display_scale = 0.5
    display_frame = cv2.resize(frame, (0,0), fx=display_scale, fy=display_scale)
    cv2.imshow("Warped View: Anomaly + YOLO Detection", display_frame)

    # --- Real-Time Syncing ---
    loop_end_time = time.time()
    process_time = loop_end_time - loop_start_time
    target_frame_time = 1.0 / 30.0
    
    wait_time_ms = 0
    if not paused:
        if process_time < target_frame_time:
            wait_time_ms = int((target_frame_time - process_time) * 1000)
            wait_time_ms = max(1, wait_time_ms)
        else:
            wait_time_ms = 1

    delay = 0 if paused else wait_time_ms
    key = cv2.waitKey(delay) & 0xFF

    actual_elapsed_time = time.time() - loop_start_time

    if key == ord('q'):
        break
    elif key == ord('p') or key == 32: 
        paused = not paused
        if not paused:
            current_frame_idx += 1
    elif key == ord('d'):
        step = 1 if paused else 30
        current_frame_idx = min(current_frame_idx + step, total_frames - 1)
    elif key == ord('a'):
        step = 1 if paused else 30
        current_frame_idx = max(current_frame_idx - step, 0)
    else:
        if not paused:
            frames_to_advance = max(1, round(actual_elapsed_time * 30.0))
            current_frame_idx += frames_to_advance

cap.release()
cv2.destroyAllWindows()
