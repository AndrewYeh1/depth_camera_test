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
        "You are a security monitor for a bicycle storage system. Analyze the image. "
        "If you see any object that is NOT a bicycle that is INSIDE of the system (e.g., a person, a bag, a scooter, a box, or foreign objects), output 'Yes' and what item you see. "
        "If the system contains ONLY the bicycle or is empty, output 'No'. "
        "It is OK for there to be people or other objects inside of the fram as long as they are not inside of the storage system."
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
            
            # Parse the text carefully to get dynamic content
            vlm_text_lower = vlm_text.lower().strip()
            
            if vlm_text_lower.startswith("yes"):
                short_text = vlm_text if len(vlm_text) < 80 else vlm_text[:77] + "..."
                # Replace newlines with spaces so it renders politely on one line
                short_text = short_text.replace("\n", " ").strip()
                last_vlm_result = f"ANOMALY: {short_text} ({inference_time:.1f}s)"
                last_color = (0, 0, 255) # Red
            elif vlm_text_lower.startswith("no"):
                last_vlm_result = f"CLEAR: NO ({inference_time:.1f}s)"
                last_color = (0, 255, 0) # Green
            else:
                short_text = vlm_text if len(vlm_text) < 80 else vlm_text[:77] + "..."
                short_text = short_text.replace("\n", " ").strip()
                last_vlm_result = f"VLM Output: {short_text} ({inference_time:.1f}s)"
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
