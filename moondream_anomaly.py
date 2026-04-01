import cv2
import ollama
import time

# Options
VIDEO_PATH = "WIN_20260326_10_11_43_Pro.mp4"
MODEL_NAME = "moondream:latest"
PROCESS_EVERY_N_FRAMES = 30 # Process 1 frame every ~1 second (assuming 30fps)

def analyze_frame_with_vlm(frame):
    """Sends a smaller version of the frame to Moondream via Ollama."""
    # Resize the frame while keeping the correct aspect ratio (don't squash it!)
    h, w = frame.shape[:2]
    scale = 512 / max(h, w) # Use 512 for a bit more clarity
    small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    # Compress frame to JPEG
    _, buffer = cv2.imencode('.jpg', small_frame)
    image_bytes = buffer.tobytes()

    prompt = (
        "Question: Look closely at the bicycle lock system. "
        "Name any object (like a backpack, bag, box, or person) that is attached to, resting on, or touching the bicycle. "
        "Keep your answer very short, just the name of the object. "
        "If the bicycle is completely bare with nothing attached to it, output exactly 'None'.\n\n"
        "Answer:"
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }],
            options={
                'temperature': 0.1 # Bumped slightly from 0.0 to prevent token collapse loops
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return "Error"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video {VIDEO_PATH}")
        return

    frame_count = 0
    last_vlm_result = "Analyzing..."
    last_vlm_time = time.time()
    last_color = (255, 255, 255)
    
    print(f"Starting VLM Anomaly Detection using {MODEL_NAME}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: rotate frame 90 degrees right like the previous script
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        display_frame = frame.copy()

        # Only run the heavy VLM on a subset of frames (or the video will freeze for 2 seconds every frame)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_time = time.time()
            vlm_text = analyze_frame_with_vlm(frame)
            inference_time = time.time() - start_time
            
            # Show the raw VLM output directly
            raw_text = vlm_text.replace("\n", " ").strip()
            if len(raw_text) > 130:
                raw_text = raw_text[:127] + "..."
                
            last_vlm_result = f"{raw_text} ({inference_time:.1f}s)"
            last_color = (0, 255, 255) # Yellow

        # Draw the LLM's last decision onto the video
        cv2.putText(display_frame, last_vlm_result, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
        
        # Display the result
        # Note: the video will physically pause every time the VLM analyzes a frame because this is synchronous.
        cv2.imshow("VLM Anomaly Detection", display_frame)

        frame_count += 1
        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
