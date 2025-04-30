# YOLO Web Application
Flask YOLOv8 Training and Inference Web App
===========================================

This project provides a web-based interface to:
- Upload a YOLOv8-formatted dataset (.zip file)
- Train YOLOv8 models (detection, segmentation, or classification)
- Run image predictions using the trained model

-----------------------------------------------------
1. Project Structure
-----------------------------------------------------

project-root/
│
├── app/
│   ├── templates/        # HTML frontend (index.html)
│   └── static/           # Static files for web interface
│
├── scripts/
│   ├── train_model.py    # Training logic using Ultralytics YOLOv8
│   └── predict.py        # Inference logic
│
├── projects/
│   └── <project_name>/
│       ├── data/         # Uploaded dataset (images and labels)
│       ├── models/       # Trained model saved here
│       └── results/      # Prediction results
│
├── app.py                # Main Flask app
└── README.txt            # Project information

-----------------------------------------------------
2. Installation
-----------------------------------------------------

Make sure you have Python 3.8 or later installed.

Install the required packages:

pip install -r requirements.txt

Sample requirements.txt:
    flask
    pyyaml
    ultralytics

-----------------------------------------------------
3. Usage Instructions
-----------------------------------------------------

Step 1: Start the Web App
--------------------------
Run the Flask application:

python app.py

Then open your browser and visit:
http://localhost:5000

Step 2: Create Project and Upload Dataset
------------------------------------------
- Enter a project name and description
- Choose task type (detect / segment / classify)
- Upload your YOLO dataset as a .zip file
- The system will automatically extract and organize the data

Expected Dataset ZIP Structure:

    dataset.zip
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

YOLO label format:
    <class_id> <x_center> <y_center> <width> <height>

Step 3: Train Your Model
--------------------------
- Enter training configuration:
    - Version (e.g. yolov8n)
    - Epochs, batch size, learning rate, image size
- Click "Train"

Trained model will be saved at:
    projects/<project_name>/models/yolov8n_trained.pt

Step 4: Predict on New Images
-------------------------------
- Upload an image (JPG, PNG)
- Output image with bounding boxes will be shown

Saved to:
    projects/<project_name>/results/

-----------------------------------------------------
4. Notes
-----------------------------------------------------

- If your ZIP file contains folders like train2017, they are automatically renamed to train/val.
- A data.yaml file is generated automatically.
- If the training folder is not detected correctly, check your dataset structure before uploading.

-----------------------------------------------------

