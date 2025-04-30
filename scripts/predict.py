# # scripts/predict.py
# import sys
# import cv2
# from yolov5 import YOLOv5

# def make_prediction(model_path, image_path, task_type):
#     # Load the YOLOv5 model
#     model = YOLOv5(model_path)
    
#     # Read the image
#     image = cv2.imread(image_path)
    
#     # Make prediction
#     results = model.predict(image)
    
#     # Handle different task types
#     if task_type == 'Detection':
#         # Display bounding boxes for detection
#         for detection in results:
#             cv2.rectangle(image, (detection['x1'], detection['y1']), (detection['x2'], detection['y2']), (0, 255, 0), 2)
#         cv2.imwrite('output_image.jpg', image)
#     elif task_type == 'Classification':
#         print(f"Predicted Class: {results[0]['label']}")
#     elif task_type == 'Segmentation':
#         mask = results[0]['segmentation_mask']
#         cv2.imwrite('segmented_image.jpg', mask)

# if __name__ == "__main__":
#     # Get model path, image path, and task type from command line arguments
#     make_prediction(sys.argv[1], sys.argv[2], sys.argv[3])
# scripts/predict.py
# scripts/predict.py
import os
import cv2
from ultralytics import YOLO

def make_prediction(model_path, image_filename, task_type, project_path):
    # Load the YOLO model
    model = YOLO(model_path)

    # Construct image path
    image_path = os.path.join(project_path, 'data', image_filename)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image from {image_path}")
        return None

    # Run prediction
    try:
        results = model(image_path)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None

    # Prepare result folder
    result_path = os.path.join(project_path, 'results')
    os.makedirs(result_path, exist_ok=True)

    # Handle different task types
    if task_type.lower() == 'detection':
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = os.path.join(result_path, f"predicted_{image_filename}")
        cv2.imwrite(output_path, image)
        return output_path

    elif task_type.lower() == 'classification':
        predicted_class = results[0].probs.argmax()
        class_file = os.path.join(result_path, "classification_result.txt")
        with open(class_file, "w") as f:
            f.write(f"Predicted Class Index: {predicted_class}\n")
        return class_file

    elif task_type.lower() == 'segmentation':
        mask = results[0].masks.data[0].cpu().numpy() * 255
        output_path = os.path.join(result_path, f"segmented_{image_filename}")
        cv2.imwrite(output_path, mask)
        return output_path

    print("[ERROR] Unsupported task type or failed to process.")
    return None
