import cv2
import os
import glob

# Directory to save all the extracted frames
output_dir = "extracted_frames_dataset"
os.makedirs(output_dir, exist_ok=True)

# Find all video files in the current directory (add more extensions if needed)
video_files = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov")

if not video_files:
    print("No video files found in the current directory.")
    exit()

print(f"Found {len(video_files)} video(s). Starting extraction...\n")

total_frames_saved = 0

for video_path in video_files:
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  -> Error: Could not open {video_path}")
        continue
        
    # Get the Frames Per Second (fps) to know how many frames make up 1 second
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print(f"  -> Warning: Invalid video properties for {video_path}")
        continue
        
    # Extract the base name of the video without extension to use in filenames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    seconds_saved = 0
    
    while True:
        # Calculate the exact frame index for the next second mark
        target_frame_idx = int(seconds_saved * fps)
        
        if target_frame_idx >= total_frames:
            # We have passed the end of the video
            break
            
        # Jump directly to that exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break # Reached end of video or read error
            
        # Scale down for display so it fits on screen
        h, w = frame.shape[:2]
        scale = min(800.0 / max(h, w), 1.0)
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        cv2.putText(display_frame, f"Video: {video_name} @ {seconds_saved}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"'s' save | ANY OTHER KEY skip | 'q' quit", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Dataset Extractor", display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting script early as requested.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
        elif key == ord('s'):
            # Construct filename: e.g. WIN_20260326_10_01_55_sec0000.png
            filename = f"{video_name}_sec{seconds_saved:04d}.png"
            save_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(save_path, frame) # Save the original full-res frame
            total_frames_saved += 1
            print(f"  -> Saved: {filename}")
        else:
            print(f"  -> Skipped second {seconds_saved}")
            
        seconds_saved += 1
        
    print(f"  -> Saved {seconds_saved} frames (1 per second) from this video.")
    cap.release()

print(f"\nExtraction complete! Saved {total_frames_saved} total frames to the '{output_dir}' folder.")
