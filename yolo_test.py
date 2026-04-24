import cv2
import json
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolo26m.pt')

    # Load ground truth from Label Studio
    json_path = 'label.json'
    try:
        with open(json_path, 'r') as f:
            ls_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find ground truth file {json_path}")
        return

    # Change this ID to select which video's labels to use (e.g., 1 or 2)
    target_task_id = 1

    # Parse JSON to build frame_to_label mapping
    frame_to_label = {}
    target_data = next((item for item in ls_data if item.get('id') == target_task_id), None)
    
    if target_data and 'annotations' in target_data and len(target_data['annotations']) > 0:
        results = target_data['annotations'][0].get('result', [])
        for r in results:
            labels = r.get('value', {}).get('timelinelabels', [])
            if not labels:
                continue
            label = labels[0]
            ranges = r.get('value', {}).get('ranges', [])
            for rng in ranges:
                # Label studio frame ranges
                start = rng.get('start', 1)
                end = rng.get('end', 1)
                for f_num in range(start, end + 1):
                    frame_to_label[f_num] = label

    # Open the video file
    video_path = 'videos/1.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    target_classes = [0] # We only care about person (0) for these stats
    
    stats = {
        "Person: Full": {"total": 0, "detected": 0, "conf_sum": 0.0},
        "Person: Partial": {"total": 0, "detected": 0, "conf_sum": 0.0},
        "Person: Limbs": {"total": 0, "detected": 0, "conf_sum": 0.0},
        "Person: None": {"total": 0, "detected": 0, "conf_sum": 0.0},
    }

    print("Evaluating YOLO against ground truth... Press 'q' to quit.")

    # Get the actual FPS of the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or video_fps is None:
        video_fps = 30.0 # Fallback

    frame_num = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_num += 1
        
        # Label Studio assumes 24 FPS for this video timeline.
        # We need to map the OpenCV frame (at ~30 FPS) to the Label Studio frame (at 24 FPS).
        current_time_sec = frame_num / video_fps
        ls_frame = int(current_time_sec * 24.0) + 1
        
        label = frame_to_label.get(ls_frame, "Unlabeled")
        
        # Determine if we should evaluate this frame
        should_evaluate = label in stats
        
        person_detected = False
        max_conf = 0.0
        display_frame = cv2.resize(frame, (1280, 720))
        yolo_status = "Skipped Inference"
        color = (150, 150, 150) # Gray for skipped

        if should_evaluate:
            # Run YOLOv8 inference for person only
            results = model(frame, classes=target_classes, verbose=False, device='cuda:0')
            
            # Parse results to find person and max confidence
            if len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if cls_id == 0:
                            person_detected = True
                            if conf > max_conf:
                                max_conf = conf
                                
            stats[label]["total"] += 1
            if person_detected:
                stats[label]["detected"] += 1
                stats[label]["conf_sum"] += max_conf
                
            annotated_frame = results[0].plot()
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            
            yolo_status = "Detected" if person_detected else "Not Detected"
            color = (0, 255, 0) if person_detected else (0, 0, 255)

        # --- UI Rendering ---
        # Display details
        cv2.putText(display_frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Truth: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if should_evaluate:
            cv2.putText(display_frame, f"YOLO: {yolo_status} ({max_conf*100:.1f}%)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(display_frame, f"YOLO: {yolo_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("YOLO Ground Truth Evaluation", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nEvaluation interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*85)
    print(f"| {'Label':<15} | % person detected | % no person detected | confidence % (detected) |")
    print("-" * 85)
    for label in ["Person: Full", "Person: Partial", "Person: Limbs", "Person: None"]:
        t = stats[label]["total"]
        if t == 0:
            print(f"| {label:<15} | {'N/A':>17} | {'N/A':>20} | {'N/A':>23} |")
            continue
            
        d = stats[label]["detected"]
        c_sum = stats[label]["conf_sum"]
        
        det_pct = (d / t) * 100
        no_det_pct = ((t - d) / t) * 100
        conf_pct = (c_sum / d) * 100 if d > 0 else 0.0
        
        print(f"| {label:<15} | {det_pct:>16.2f}% | {no_det_pct:>19.2f}% | {conf_pct:>22.2f}% |")
    print("="*85 + "\n")

if __name__ == '__main__':
    main()
