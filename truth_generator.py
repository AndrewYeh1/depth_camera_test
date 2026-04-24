import cv2
import json

def main():
    video_path = 'videos_1/black bike no light.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    person_state_list = ["full", "partial", "limbs", "none"]
    bike_state_list = ["full", "partial", "none"]
    person_idx = 3 # default to "none"
    bike_idx = 2 # default to "none"
    person_frame_state = {}
    bike_frame_state = {}

    # Read first frame to show while initially paused
    success, frame = cap.read()
    if not success:
        print("Failed to read the first frame.")
        return
    clean_frame = frame.copy()
    frame_num = 1
    paused = True
    playback_delay = 30

    print("Controls:")
    print("  'p' or Spacebar : Pause/Resume")
    print("  '[' / ']' : Decrease/Increase Playback Speed")
    print("  '1' : Cycle Person State")
    print("  '2' : Cycle Bike State")
    print("  'q' : Quit")

    while cap.isOpened():
        if not paused:
            success, frame = cap.read()
            if not success:
                break
            clean_frame = frame.copy()
            frame_num += 1
            person_frame_state[frame_num] = person_idx
            bike_frame_state[frame_num] = bike_idx
        
        display_frame = clean_frame.copy()
        person_state = f"Person: {person_state_list[person_idx]}"
        bike_state = f"Bike: {bike_state_list[bike_idx]}"
        
        cv2.putText(display_frame, person_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, bike_state, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Delay: {playback_delay} ms", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Truth Generator", display_frame)
        
        key = cv2.waitKey(0 if paused else playback_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p") or key == ord(" "):
            paused = not paused
        elif key == ord("]"):
            playback_delay = max(1, playback_delay - 10)
        elif key == ord("["):
            playback_delay += 10
        elif paused:
            if key == ord("1"):
                person_idx = (person_idx + 1) % len(person_state_list)
            elif key == ord("2"):
                bike_idx = (bike_idx + 1) % len(bike_state_list)
    
    with open("truth.json", "w") as f:
        json.dump({
            "person": person_frame_state,
            "bike": bike_frame_state
        }, f)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
