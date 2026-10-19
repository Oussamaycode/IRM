# 🧠 Brain MRI Classification and Tumor Detection

## 📌 Project Overview

This project aims to develop a Deep Learning prototype for **classifying brain MRI images and detecting tumor regions** using Transfer Learning and Object Detection techniques.

The system combines a pre-trained CNN model for image classification with YOLOv8 for tumor localization. It also integrates Airflow for pipeline automation, Streamlit for the user interface, and Docker for containerization.

The project is designed for educational and experimental purposes to explore Computer Vision, Deep Learning, and MLOps workflows.

**⚠️ Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.

## 🎯 Objectives

- Classify brain MRI images into four categories.
- Detect and localize annotated tumor regions.
- Apply Transfer Learning using pre-trained CNN architectures.
- Reduce overfitting through regularization and training callbacks.
- Evaluate classification and object detection performance.
- Automate the machine learning workflow using Apache Airflow.
- Build an interactive interface using Streamlit.
- Containerize the application using Docker.

## 📂 Dataset

**Dataset:** [MRI for Brain Tumor with Bounding Boxes — Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

The dataset contains brain MRI images belonging to four categories:

| Class | Description |
|---|---|
| Glioma | MRI images associated with glioma tumors |
| Meningioma | MRI images associated with meningioma tumors |
| No Tumor | MRI images without an annotated tumor |
| Pituitary | MRI images associated with pituitary tumors |

The dataset also includes bounding box annotations that can be used to train and evaluate an object detection model.

### Data Preparation

The dataset preparation pipeline includes:

- Exploring class distribution and image dimensions.
- Visualizing images and their annotations.
- Detecting corrupted files and inconsistent bounding boxes.
- Verifying annotation formats and coordinate systems.
- Splitting the data into training, validation, and test sets.
- Preventing data leakage between the different sets.
- Converting bounding boxes into YOLO format when necessary.

**Important:** The annotations for the `No Tumor` class must be inspected carefully. Images without tumor regions should remain available for classification. For object detection, only valid, annotated tumor regions should be used as ground-truth objects. Images without objects must be represented correctly in the YOLO dataset.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep Learning and image classification |
| MobileNetV2 / ResNet50 | Transfer Learning architectures |
| Ultralytics YOLOv8 | Tumor object detection |
| OpenCV | Image processing and visualization |
| NumPy / Pandas | Data manipulation and analysis |
| Matplotlib / Seaborn | Data visualization and evaluation plots |
| Scikit-learn | Classification metrics and data splitting |
| Apache Airflow | Pipeline automation and orchestration |
| Streamlit | Interactive web application |
| Docker / Docker Compose | Containerization and service management |
| Git / GitHub | Version control and project collaboration |

## 🧠 Project Workflow

The project consists of two main Machine Learning tasks.

### 1. Brain MRI Classification

Transfer Learning is used to classify an MRI image into one of four categories.

The classification workflow includes:

1. Load and preprocess MRI images.
2. Resize images to the input dimensions required by the selected model.
3. Apply data augmentation to training images.
4. Load a CNN model pre-trained on ImageNet.
5. Freeze the pre-trained layers and train a new classification head.
6. Unfreeze selected layers for fine-tuning.
7. Evaluate the model on validation and test datasets.

Candidate architectures:

- **MobileNetV2:** A lightweight architecture suitable for efficient inference.
- **ResNet50:** A deeper architecture that can be used for performance comparison.

### 2. Tumor Detection with YOLOv8

YOLOv8 is used to identify and localize annotated tumor regions in MRI images.

The detection workflow includes:

1. Inspect and validate bounding box annotations.
2. Convert annotations into YOLO format.
3. Configure the dataset and class definitions.
4. Load a pre-trained YOLOv8 detection model.
5. Train the model on the prepared dataset.
6. Evaluate detection performance using IoU and mAP.
7. Visualize predicted bounding boxes and confidence scores.
8. Analyze false positives and missed tumor regions.

**Detection scope:** The model should detect only regions that are correctly annotated. The four-class image classification problem and the object detection problem may therefore use different label definitions.

## 🛡️ Overfitting Prevention

Several techniques are used to improve generalization and reduce overfitting.

- **Data Augmentation:** Applies moderate rotations, translations, zooms, and contrast adjustments to training images.
- **Dropout:** Randomly disables neurons during training to reduce dependence on specific features.
- **L2 Regularization:** Penalizes excessively large model weights.
- **Early Stopping:** Stops training when validation performance no longer improves.
- **ModelCheckpoint:** Saves the best-performing model based on a validation metric.
- **ReduceLROnPlateau:** Reduces the learning rate when validation performance stagnates.
- **Fine-Tuning:** Unfreezes selected pre-trained layers using a lower learning rate.

At least two training configurations should be compared to evaluate the impact of these techniques.

## 📊 Model Evaluation

### Classification Metrics

The classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation learning curves
- Error analysis by class

MobileNetV2 and ResNet50 can be compared using the same data split and evaluation procedure.

### Object Detection Metrics

The YOLOv8 model is evaluated using:

- **IoU (Intersection over Union):** Measures the overlap between predicted and ground-truth bounding boxes.
- **mAP@0.5:** Mean Average Precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** Mean Average Precision across multiple IoU thresholds.
- Precision and recall.
- Visual inspection of predicted bounding boxes.
- Analysis of false positives and missed detections.

The validation set is used for model selection. The test set is reserved for the final evaluation.

## ⚙️ Automated Pipeline with Apache Airflow

Apache Airflow orchestrates the main steps of the machine learning workflow through a Directed Acyclic Graph (DAG).

The pipeline includes the following tasks:

1. **Data Validation:** Check image files, labels, and annotations.
2. **Preprocessing:** Prepare images and bounding boxes.
3. **Model Training:** Train the classification and detection models.
4. **Evaluation:** Calculate metrics and generate evaluation reports.
5. **Artifact Saving:** Store trained models, metrics, and relevant outputs.

Task dependencies ensure that training starts only after data preparation succeeds and that evaluation follows training.

The pipeline should also include error handling and logging to facilitate debugging.

## 🖥️ Streamlit Application

The Streamlit interface provides an interactive way to test the trained models.

### Main Features

- Upload a brain MRI image.
- Display the uploaded image.
- Predict one of the four classification categories.
- Display the predicted class probabilities.
- Visualize detected tumor regions.
- Display bounding boxes and detection confidence scores.
- Consult model evaluation metrics and training results.

Trained models are loaded from saved artifacts so that model training does not run again whenever a user interacts with the application.

## 🐳 Docker and Containerization

Docker is used to package the application and its dependencies into reproducible environments.

Docker Compose can orchestrate the following services:

- **Streamlit:** Runs the interactive application.
- **Airflow:** Schedules and executes the machine learning pipeline.
- **PostgreSQL:** Stores Airflow metadata if configured as the metadata database.

Shared volumes can be used to persist datasets, trained models, logs, and evaluation results.

## 📁 Project Structure

```text
brain-mri-classification-detection/
│
├── dags/
│   └── .gitkeep
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   │   └── .gitkeep
│       │   ├── val/
│       │   │   └── .gitkeep
│       │   └── test/
│       │       └── .gitkeep
│       ├── labels/
│       │   ├── train/
│       │   │   └── .gitkeep
│       │   ├── val/
│       │   │   └── .gitkeep
│       │   └── test/
│       │       └── .gitkeep
│       └── .gitkeep
│
├── models/
│   ├── classification/
│   │   └── .gitkeep
│   └── detection/
│       └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── src/
│   ├── data/
│   │   └── .gitkeep
│   ├── classification/
│   │   └── .gitkeep
│   ├── detection/
│   │   └── .gitkeep
│   └── utils/
│       └── .gitkeep
│
├── app/
│   └── .gitkeep
│
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
│
├── tests/
│   └── .gitkeep
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── Readme.md
```

The directories are scaffolded with `.gitkeep` files so Git can track them while they are empty. Dataset files and trained model artifacts are not committed. Add implementation files as each project stage is completed.

## 🚀 Installation and Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.10 or a compatible version supported by the selected libraries.
- Git.
- Docker and Docker Compose.
- A Kaggle account to download the dataset.

A compatible NVIDIA GPU can accelerate training, but CPU execution may be possible with longer training times.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd brain-mri-classification-detection
```

Replace `<YOUR_REPOSITORY_URL>` with the URL of your Git repository.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The dependency file should contain the packages required for data processing, TensorFlow, Ultralytics, Streamlit, Airflow, evaluation, and testing. Airflow should be installed using the official installation instructions and compatible constraints for the selected Python version.

### 4. Download the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes

Extract the files into the `data/raw/` directory and adapt the preprocessing scripts to the dataset's actual directory structure and annotation format.

### 5. Prepare the Data

Run the data validation and preprocessing scripts:

```bash
python src/data/validate_data.py
python src/data/preprocess.py
```

These commands assume the scripts have been implemented with the corresponding command-line entry points.

### 6. Train the Classification Model

```bash
python src/classification/train.py
```

### 7. Train the YOLOv8 Detection Model

```bash
python src/detection/train.py
```

The training scripts should save the trained models and associated configuration files in the `models/` directory.

### 8. Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal to access the application.

## 🔄 Running with Docker Compose

Build and start the configured services:

```bash
docker compose up --build -d
```

Inspect the running containers:

```bash
docker compose ps
```

View the service logs:

```bash
docker compose logs -f
```

Stop the services:

```bash
docker compose down
```

The `Dockerfile` and `docker-compose.yml` must define the required dependencies, environment variables, volumes, service dependencies, and startup commands.

Airflow may require additional initialization steps and configuration before its services can run successfully. Follow the official Airflow Docker deployment documentation when configuring the environment.

## 🧪 Testing

Run the automated tests with:

```bash
pytest tests/
```

Tests should cover:

- Image loading and preprocessing.
- Detection annotation validity.
- Bounding box coordinate conversion.
- Data split integrity and leakage prevention.
- Classification output dimensions and probability values.
- Detection output format and bounding box validity.
- Model loading and inference.
- Pipeline task failures and dependency handling.

## 📈 Expected Deliverables

- Dataset exploration and data quality report.
- Cleaned and prepared classification dataset.
- YOLO-formatted detection dataset.
- Trained classification models and comparison results.
- Trained YOLOv8 model.
- Classification and detection evaluation reports.
- Confusion matrix and learning curves.
- Detection visualizations and error analysis.
- Airflow DAG for workflow automation.
- Streamlit web application.
- Docker configuration.
- Automated tests and project documentation.

Actual performance metrics should be added after training and evaluating the models. No results are assumed in advance.

## 📅 Project Timeline — 10 Days

| Day | Tasks |
|---|---|
| Day 1 | Explore the dataset and inspect annotations |
| Day 2 | Preprocess images, split datasets, and implement augmentation |
| Day 3 | Train the initial MobileNetV2 classification model |
| Day 4 | Apply regularization, fine-tuning, and classification evaluation |
| Day 5 | Prepare YOLO annotations and configure the detection pipeline |
| Day 6 | Train YOLOv8 and analyze detection performance |
| Day 7 | Automate the workflow with Apache Airflow |
| Day 8 | Develop the Streamlit application |
| Day 9 | Integrate Docker and run tests |
| Day 10 | Complete documentation and prepare the final demonstration |

## 🔮 Future Improvements

- Compare additional pre-trained CNN architectures.
- Experiment with different YOLOv8 model sizes.
- Optimize inference speed and memory consumption.
- Add more detailed visualization of model predictions.
- Track experiments and metrics across training runs.
- Add model versioning and reproducible training configurations.
- Improve data validation and automated testing.
- Explore explainability techniques such as Grad-CAM for classification.

## 📚 References

- [MRI for Brain Tumor with Bounding Boxes — Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

## 👨‍💻 Authors

Developed as an educational Deep Learning and MLOps project.

**Contributions:** Contributions, suggestions, and improvements are welcome.

---

**Disclaimer:** This project is a learning prototype. Its predictions are not validated for clinical use and must not replace professional medical evaluation.
