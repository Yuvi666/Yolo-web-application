from flask import Flask, render_template, request, redirect, url_for, flash
import os
import zipfile
import yaml
import shutil
from scripts.train_model import train_yolo_model
from scripts.predict import make_prediction

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'your_secret_key'
BASE_PROJECT_PATH = 'projects/'

def create_data_yaml(dataset_dir):
    # Initialize class names as 'class0' by default
    names = ['class0']
    label_train_path = os.path.join(dataset_dir, 'labels', 'train')

    # Get class names from the training labels
    if os.path.exists(label_train_path):
        class_ids = set()
        for fname in os.listdir(label_train_path):
            if fname.endswith('.txt'):
                with open(os.path.join(label_train_path, fname), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_ids.add(int(line.split()[0]))  # Extract class IDs from labels
        if class_ids:
            names = [f'class{i}' for i in sorted(class_ids)]

    # Set up paths for images and labels (assuming 'train2017' exists)
    train_path = os.path.abspath(os.path.join(dataset_dir, 'images', 'train')).replace('\\', '/')

    # If the dataset does not have 'val', use the same folder for validation
    val_path = os.path.abspath(os.path.join(dataset_dir, 'images', 'train')).replace('\\', '/')

    # Ensure the 'train2017' directory exists
    if not os.path.exists(train_path):
        raise ValueError("Training images directory 'train2017' not found.")
    
    # Prepare the data.yaml content
    data_yaml = {
        'train': train_path,
        'val': val_path,  # Using the same directory for val
        'nc': len(names),  # Number of classes
        'names': names      # List of class names
    }

    # Save the YAML file to dataset directory
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

def extract_and_organize_dataset(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Handle nested structure: Extract the contents if the dataset is inside a sub-folder
    extracted_items = os.listdir(extract_dir)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
        nested_root = os.path.join(extract_dir, extracted_items[0])
        for item in os.listdir(nested_root):
            shutil.move(os.path.join(nested_root, item), extract_dir)
        os.rmdir(nested_root)

    # Rename 'train2017' and 'val2017' if present
    for root, dirs, _ in os.walk(extract_dir):
        if 'train2017' in dirs:
            os.rename(os.path.join(root, 'train2017'), os.path.join(root, 'train'))
        if 'val2017' in dirs:
            os.rename(os.path.join(root, 'val2017'), os.path.join(root, 'val'))

    images_train = os.path.join(extract_dir, 'images', 'train')
    labels_train = os.path.join(extract_dir, 'labels', 'train')

    # Ensure required directories for training data exist
    required_dirs = [images_train, labels_train]
    if not all(os.path.exists(d) for d in required_dirs):
        return False

    # If 'val' folder does not exist, create it from 'train' images
    images_val = os.path.join(extract_dir, 'images', 'val')
    labels_val = os.path.join(extract_dir, 'labels', 'val')

    if not os.path.exists(images_val):
        os.makedirs(images_val)
        for image in os.listdir(images_train):
            shutil.copy(os.path.join(images_train, image), images_val)
    
    if not os.path.exists(labels_val):
        os.makedirs(labels_val)
        for label in os.listdir(labels_train):
            shutil.copy(os.path.join(labels_train, label), labels_val)

    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        project_name = request.form['project_name'].strip()
        description = request.form['description'].strip()
        task_type = request.form['task_type']
        dataset_zip = request.files['dataset']

        project_dir = os.path.join(BASE_PROJECT_PATH, project_name)
        if os.path.exists(project_dir):
            flash("Project already exists.", "danger")
            return redirect(url_for('index'))

        os.makedirs(os.path.join(project_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'results'), exist_ok=True)

        with open(os.path.join(project_dir, 'description.txt'), 'w') as f:
            f.write(description)

        if dataset_zip.filename.endswith('.zip'):
            zip_path = os.path.join(project_dir, 'dataset.zip')
            dataset_zip.save(zip_path)

            if not extract_and_organize_dataset(zip_path, os.path.join(project_dir, 'data')):
                flash("Dataset is missing required structure: images/train, images/val, labels/train, labels/val", "danger")
                shutil.rmtree(project_dir)
                return redirect(url_for('index'))

            os.remove(zip_path)
            create_data_yaml(os.path.join(project_dir, 'data'))

            flash("Project created and dataset uploaded successfully.", "success")
            return render_template('index.html', created=True, project=project_name, task=task_type)

    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    project_name = request.form['project']
    task_type = request.form['task']
    version = request.form['version']
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    learning_rate = float(request.form['learning_rate'])
    image_size = int(request.form['image_size'])

    project_dir = os.path.join(BASE_PROJECT_PATH, project_name)
    data_yaml_path = os.path.join(project_dir, 'data', 'data.yaml')
    train_yolo_model(version, task_type, epochs, batch_size, learning_rate, image_size, project_dir, data_yaml_path)
    flash("Model training started/completed.", "info")
    return render_template('index.html', created=True, trained=True, project=project_name, task=task_type)

@app.route('/predict', methods=['POST'])
def predict():
    project_name = request.form['project']
    task_type = request.form['task']
    image = request.files['image']

    project_dir = os.path.join(BASE_PROJECT_PATH, project_name)
    image_path = os.path.join(project_dir, 'data', image.filename)
    image.save(image_path)

    result_image_path = make_prediction(
        model_path=os.path.join(project_dir, 'models', 'yolov8n_trained.pt'),
        image_filename=image.filename,
        task_type=task_type,
        project_path=project_dir
    )

    # Handle failure
    if result_image_path is None or not os.path.exists(result_image_path):
        flash("Prediction failed. Please check the model or image format.", "danger")
        return redirect(url_for('index'))

    # If result is an image, show it. If it's text, don’t try to display as image.
    result_img = None
    if result_image_path.endswith(('.jpg', '.jpeg', '.png')):
        result_img = result_image_path.replace("app/static/", "")  # Adjust this path if needed

    return render_template('index.html', created=True, trained=True, predicted=True,
                           project=project_name, task=task_type,
                           result_img=result_img)

if __name__ == '__main__':
    app.run(debug=True)
