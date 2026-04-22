from ultralytics import YOLO
import cv2
import json
from collections import Counter

# -----------------------------
# 1. Load YOLO Model
# -----------------------------
model = YOLO("yolov8n.pt")   # you can use yolov8m.pt for better accuracy

# -----------------------------
# 2. Load Image
# -----------------------------
image_path = r"C:\data\camera.jpeg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

# -----------------------------
# 3. Perform Object Detection
# -----------------------------
results = model(img, imgsz=200)   # smaller size = faster

# -----------------------------
# 4. Extract Detected Objects
# -----------------------------
class_ids = results[0].boxes.cls.tolist()
class_names = [model.names[int(i)] for i in class_ids]

# Count objects
counts = Counter(class_names)

# -----------------------------
# 5. Prepare JSON Output
# -----------------------------
output_data = {
    "image": image_path,
    "total_objects": sum(counts.values()),
    "object_counts": dict(counts)
}

# -----------------------------
# 6. Print JSON Output
# -----------------------------
print(json.dumps(output_data, indent=4))

# -----------------------------
# 7. Save JSON to File
# -----------------------------
with open("output.json", "w") as f:
    json.dump(output_data, f, indent=4)

# -----------------------------
# 8. Get Annotated Image
# -----------------------------
annotated_img = results[0].plot()

# -----------------------------
# 9. Resize Image (Fix Large Size)
# -----------------------------
h, w = annotated_img.shape[:2]

max_width = 500  # change size here (600 / 1000 etc.)
scale = max_width / w

new_w = int(w * scale)
new_h = int(h * scale)

resized_img = cv2.resize(annotated_img, (new_w, new_h))

# -----------------------------
# 10. Save Compressed Image
# -----------------------------
cv2.imwrite("output.jpg", resized_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
  
# -----------------------------
# 11. Show Image
# -----------------------------
cv2.imshow("Detected Objects", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()