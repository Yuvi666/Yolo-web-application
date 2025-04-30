from ultralytics import YOLO
import os

def train_yolo_model(version, task_type, epochs, batch_size, learning_rate, image_size, project_path,data_yaml_path):
    # Initialize YOLO model with the selected version, like 'yolov8n.pt'
    data_yaml_path = data_yaml_path.replace('\\', '/')
    print("[DEBUG] Using data.yaml at:", data_yaml_path)
    model = YOLO(f'{version}.pt')  # Load YOLO model from the provided version

    # Data YAML path (required for training)
   
    # data_yaml_path = os.path.join(project_path, 'data.yaml')

    # Start training with the model, providing task-related data
    model.train(
        data=data_yaml_path,  # Path to data.yaml
        epochs=epochs,
        imgsz=image_size,      # Image size (input resolution)
        batch=batch_size,      # Batch size for training
        lr0=learning_rate,    # Learning rate
        project=os.path.join(project_path, 'results'),  # Output path
        name='train'  # Folder name under 'results/'
    )

    # Save the trained model in the models folder
    model.save(os.path.join(project_path, 'models', f'{version}_trained.pt'))
    print("[INFO] Training complete and model saved.")
