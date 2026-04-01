import cv2
import ollama
import time

# Options
VIDEO_PATH = "WIN_20260326_10_11_43_Pro.mp4"
MODEL_NAME = "moondream:latest"
PROCESS_EVERY_N_FRAMES = 30 # Process 1 frame every ~1 second (assuming 30fps)

def analyze_frame_with_vlm(frame):
    """Sends a smaller version of the frame to Moondream via Ollama."""
    # Resize the frame to a smaller resolution (e.g. 384x384) to save VRAM and speed up processing
    small_frame = cv2.resize(frame, (384, 384))
    
    # Compress frame to JPEG
    _, buffer = cv2.imencode('.jpg', small_frame)
    image_bytes = buffer.tobytes()

    prompt = (
        "You are a security monitor for a bicycle. Analyze the image. "
        "If you see any object that is NOT a bicycle (e.g., a person, a bag, a scooter, a box, or foreign objects), output 'Yes'. "
        "If the image contains ONLY the bicycle or is empty, output 'No'. "
        "Do not provide any other text."
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
                'temperature': 0.0 # Force deterministic output
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
            
            # Clean up the response just in case moondream adds punctuation
            vlm_text_clean = "".join(c for c in vlm_text if c.isalpha()).lower()
            
            if "yes" in vlm_text_clean:
                last_vlm_result = f"ANOMALY: YES ({inference_time:.1f}s)"
                color = (0, 0, 255) # Red
            elif "no" in vlm_text_clean:
                last_vlm_result = f"CLEAR: NO ({inference_time:.1f}s)"
                color = (0, 255, 0) # Green
            else:
                last_vlm_result = f"UNKNOWN: {vlm_text} ({inference_time:.1f}s)"
                color = (0, 255, 255) # Yellow

        # Draw the LLM's last decision onto the video
        color = (0, 0, 255) if "YES" in last_vlm_result else ((0, 255, 0) if "NO" in last_vlm_result else (0, 255, 255))
        cv2.putText(display_frame, last_vlm_result, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
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
