import cv2
from ultralytics import YOLO
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on device: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
model = YOLO("yolo26m-seg.pt")
model.to(device)  # Explicitly move model to selected device

cap = cv2.VideoCapture("WIN_20260326_10_01_55_Pro.mp4")

# --- Load the reference frame for background subtraction ---
reference = cv2.imread('reference.png')
if reference is not None:
    # Make sure reference is matched to the runtime dimensions & orientation
    if reference.shape[:2] != (640, 384): # (Height, Width)
        reference = cv2.resize(reference, (384, 640))
    
    # Blur the full color reference image instead of grayscale
    ref_blur = cv2.GaussianBlur(reference, (21, 21), 0)
else:
    print("Warning: reference.png not found. Foreign object detection disabled.")
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break
        
    # Rotate the frame 90 degrees to the right (clockwise)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Downscale frame to exactly match YOLO's processing resolution (Width 384 x Height 640)
    frame = cv2.resize(frame, (384, 640))
        
    # Run YOLO inference with increased sensitivity (lowered confidence threshold)
    # Default is usually 0.25. Setting it to 0.1 makes it much more likely to detect the bike.
    results = model(frame, conf=0.1)
    
    # Store polygons for the bicycle/motorcycle and any other detected object
    bike_polygons = []
    other_polygons = []
    if results[0].masks is not None:
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 1 or int(cls) == 3: # seems to detect bicycles as motorcycles a lot
                bike_polygons.append(np.array(results[0].masks.xy[i], np.int32))
            else: # ANY other class YOLO detects (Person, Backpack, Handbag, etc.)
                other_polygons.append(np.array(results[0].masks.xy[i], np.int32))

    # Create a unified mask of pixels to ignore (the bike, but EXCLUDING any other object)
    ignore_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for pts in bike_polygons:
        cv2.fillPoly(ignore_mask, [pts], 255) # Ignore bike
    for pts in other_polygons:
        cv2.fillPoly(ignore_mask, [pts], 0)   # Do NOT ignore any other detected object

    # --- Foreign Object Detection via absdiff ---
    if reference is not None:
        # Blur the raw BGR color frame (do not convert to grayscale first!)
        frame_blur = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Calculate the absolute difference across all 3 color channels (Blue, Green, Red)
        diff_color = cv2.absdiff(ref_blur, frame_blur)
        
        # Convert the color difference output into a single intensity channel so we can threshold it
        diff_intensity = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
        
        # MASK OUT THE BIKE FROM THE DIFFERENCE: 
        # This prevents the bike's actual pixels from registering as "foreign objects",
        # but since we punched a hole for the backpack, it WILL be checked.
        diff_intensity[ignore_mask == 255] = 0
            
        # Threshold the difference map to black & white
        # Increased threshold from 25 to 70 to completely ignore subtle lighting shifts and shadows,
        # forcing it to only detect drastic pixel color changes (actual physical objects).
        _, thresh = cv2.threshold(diff_intensity, 50, 255, cv2.THRESH_BINARY)
        
        # Erode first to peel away tiny pixel noise, then dilate heavily to group the solid objects
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=4)
        
        # Find contours of differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000: # Ignore tiny differences/noise
                continue
            # Draw a bounding box around large enough differences
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Foreign Obj", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # As previously requested, black out the bicycle in the display window
    # Using the ignore_mask ensures the backpack remains fully visible!
    frame[ignore_mask == 255] = (0, 0, 0)

    # Draw blue bounding boxes around all objects YOLO detected
    if results[0].boxes is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    # Optional debugging window to see what the algorithm "sees" changing:
    # if reference is not None: cv2.imshow("Difference Map", diff)

    delay = 0 if paused else 1
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p') or key == 32: # 'p' or Spacebar
        paused = not paused
        # Advance slightly if we just hit play to prevent re-detecting the same frame
        if not paused:
            current_frame_idx += 1
    elif key == ord('d'):
        step = 1 if paused else 30 # fast-forward 30 frames
        current_frame_idx = min(current_frame_idx + step, total_frames - 1)
    elif key == ord('a'):
        step = 1 if paused else 30 # rewind 30 frames
        current_frame_idx = max(current_frame_idx - step, 0)
    else:
        # Normal playback automatically advances
        if not paused:
            current_frame_idx += 1

cap.release()
cv2.destroyAllWindows()
