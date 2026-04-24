import cv2
import os

video_path = "videos/1.mp4"
output_dir = "extracted_frames_1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame_idx = 0

print(f"Video loaded: {video_path}")
print(f"Total Frames: {total_frames}")
print("\nControls:")
print("  d  : Next frame")
print("  a  : Previous frame")
print("  f  : Fast forward 10 frames")
print("  b  : Rewind 10 frames")
print("  s  : Save current frame")
print("  r  : Toggle 90-degree right rotation")
print("  q  : Quit")

rotate_90 = True  # We start true since the object_detection script had it rotated

while True:
    # Set the video position to the current frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        current_frame_idx = max(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
    # Keep the original full-res frame for saving
    save_frame = frame.copy()
    
    # Optionally rotate the frame
    if rotate_90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        save_frame = cv2.rotate(save_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    # Resize for display so it fits nicely on modern monitor screens
    h, w = frame.shape[:2]
    # Downscale for display by a factor of 2, 3 or whatever fits into roughly 1080p height
    scale = min(1200 / max(h, w), 1.0)
    
    display_w = int(w * scale)
    display_h = int(h * scale)
    
    display_frame = cv2.resize(frame, (display_w, display_h))
    
    # Put text showing current frame on the display copy
    cv2.putText(display_frame, f"Frame: {current_frame_idx} / {total_frames}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if rotate_90:
         cv2.putText(display_frame, f"Rotated 90 deg", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
         cv2.putText(display_frame, f"Original orientation", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow("Frame Extractor", display_frame)
    
    # Wait indefinitely for a key press
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('d'):       # 'd' next
        current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
    elif key == ord('a'):       # 'a' prev
        current_frame_idx = max(current_frame_idx - 1, 0)
    elif key == ord('f'):       # 'f' fast forward
        current_frame_idx = min(current_frame_idx + 10, total_frames - 1)
    elif key == ord('b'):       # 'b' rewind
        current_frame_idx = max(current_frame_idx - 10, 0)
    elif key == ord('r'):       # 'r' toggle rotation
        rotate_90 = not rotate_90
    elif key == ord('s'):       # 's' save frame
        save_path = os.path.join(output_dir, f"frame_{current_frame_idx:04d}.png")
        cv2.imwrite(save_path, save_frame)
        print(f"Saved: {save_path}")

cap.release()
cv2.destroyAllWindows()
