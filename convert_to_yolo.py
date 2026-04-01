import os
import json
import shutil

def convert_labelme_to_yolo():
    input_dir = "extracted_frames"
    dataset_dir = "dataset"
    
    # Define your classes here
    classes = {"Bicycle": 0}
    
    # Create YOLO directory structure
    images_dir = os.path.join(dataset_dir, "images", "train")
    labels_dir = os.path.join(dataset_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process each json file
    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue
            
        json_path = os.path.join(input_dir, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
        img_filename = data["imagePath"]
        img_path = os.path.join(input_dir, img_filename)
        
        # Check if the image actually exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
            
        # Copy image to dataset structure
        shutil.copy(img_path, os.path.join(images_dir, img_filename))
        
        # Create YOLO label file (.txt)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as txt_f:
            for shape in data["shapes"]:
                label = shape["label"]
                if label not in classes:
                    continue  # Ignore unknown classes
                
                class_id = classes[label]
                points = shape["points"]
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1)
                normalized_points = []
                for x, y in points:
                    norm_x = x / img_width
                    norm_y = y / img_height
                    normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                
                line = f"{class_id} " + " ".join(normalized_points)
                txt_f.write(line + "\n")
                
    # Create dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {os.path.abspath(dataset_dir)}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/train\n\n")  # Using train for val since dataset is tiny
        yf.write("names:\n")
        for cls_name, cls_id in classes.items():
            yf.write(f"  {cls_id}: {cls_name}\n")
            
    print(f"Successfully created YOLO dataset in '{dataset_dir}'")
    print(f"Dataset config written to '{yaml_path}'")

if __name__ == "__main__":
    convert_labelme_to_yolo()
