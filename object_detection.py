import cv2
from ultralytics import YOLO
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on device: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
model = YOLO("yolo11s-seg.pt")
model.to(device)  # Explicitly move model to selected device

cap = cv2.VideoCapture("WIN_20260326_10_01_55_Pro.mp4")

# --- Load the reference frame for background subtraction ---
reference = cv2.imread('reference.png')
if reference is not None:
    # Make sure reference is matched to the runtime dimensions & orientation
    if reference.shape[:2] != (640, 384): # (Height, Width)
        reference = cv2.resize(reference, (384, 640))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (21, 21), 0)
else:
    print("Warning: reference.png not found. Foreign object detection disabled.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Rotate the frame 90 degrees to the right (clockwise)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Downscale frame to exactly match YOLO's processing resolution (Width 384 x Height 640)
    frame = cv2.resize(frame, (384, 640))
        
    # Run YOLO inference
    results = model(frame)
    
    # Store polygons for the bicycle/motorcycle to mask out later
    bike_polygons = []
    if results[0].masks is not None:
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 1 or int(cls) == 3: # seems to detect bicycles as motorcycles a lot
                bike_polygons.append(np.array(results[0].masks.xy[i], np.int32))

    # --- Foreign Object Detection via absdiff ---
    if reference is not None:
        # Convert the raw frame to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate the absolute difference between background and current frame
        diff = cv2.absdiff(ref_blur, gray_blur)
        
        # MASK OUT THE BIKE FROM THE DIFFERENCE: 
        # This prevents the bike's actual pixels from registering as "foreign objects".
        for pts in bike_polygons:
            cv2.fillPoly(diff, [pts], 0)
            
        # Threshold the difference map to black & white
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
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
    for pts in bike_polygons:
        cv2.fillPoly(frame, [pts], (0, 0, 0))

    cv2.imshow("Frame", frame)
    # Optional debugging window to see what the algorithm "sees" changing:
    # if reference is not None: cv2.imshow("Difference Map", diff)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
