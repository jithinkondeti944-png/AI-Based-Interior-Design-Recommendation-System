# AI-Based Interior Design Recommendation System

An AI-powered interior design recommendation system that analyzes a room image, understands the room type, detects furniture and other objects, and recommends visually similar interiors and furniture using deep learning and similarity search.

## Overview

The system follows an end-to-end computer vision pipeline:

**Room Image → Room Classification → Furniture Detection → Feature Extraction → FAISS Similarity Search → Recommendations → E-commerce Search Link**

The project combines multiple deep learning techniques to understand both the overall room and the individual furniture present in it.

## Key Features

* Room classification using deep learning models
* Comparison of **Vision Transformer (ViT), ResNet50, and VGG16**
* Furniture and object detection using **YOLOv8**
* Visual feature extraction using pretrained **ResNet50**
* Similarity search using **FAISS**
* Room-level and furniture-level recommendations
* Dynamically generated Amazon search links

## Technologies Used

* Python
* PyTorch
* Torchvision
* Vision Transformer (ViT)
* ResNet50
* VGG16
* YOLOv8
* FAISS
* Scikit-learn
* PIL
* Matplotlib
* Kaggle Dataset

## System Architecture

### 1. Room Classification

The input room image is first classified into a room category.

The project evaluates:

* Vision Transformer (ViT)
* ResNet50
* VGG16

The models use pretrained ImageNet weights with their final classification layers replaced according to the number of room classes.

### 2. Furniture Detection

After identifying the room type, **YOLOv8m** detects furniture and relevant objects.

Examples include:

* Chair
* Sofa
* Bed
* Dining table
* TV
* Indoor plant
* Refrigerator
* Microwave
* Vase
* Clock

Detected objects are filtered based on confidence and minimum object size, and the detected objects are cropped for further processing.

### 3. Visual Feature Extraction

A pretrained **ResNet50 without its classification layer** is used as a feature extractor.

Each room image or furniture crop is converted into a **2048-dimensional visual feature vector**.

These embeddings represent the visual characteristics of the image and can be compared with other images.

### 4. FAISS Similarity Search

The generated feature vectors are normalized and stored in FAISS indexes.

The project maintains:

* Room-level FAISS indexes for similar interiors
* Furniture-level FAISS indexes for similar furniture

FAISS uses inner-product similarity on normalized vectors, which corresponds to cosine similarity.

### 5. E-commerce Links

For detected furniture, the system dynamically generates an Amazon search link based on the detected furniture type and predicted room.

For example:

`sofa + living room → Amazon search`

The generated URL is a search link rather than a specific product ID.

## Dataset

The room classification dataset used in the notebook is the **House Rooms Image Dataset** from Kaggle.

Dataset:

`robinreni/house-rooms-image-dataset`

The notebook downloads the dataset using the Kaggle API.

## Data Preprocessing

Images are:

* Resized to `224 × 224`
* Converted to tensors
* Normalized using ImageNet mean and standard deviation

Training images additionally use:

* Random horizontal flipping
* Color jitter augmentation

The dataset is divided into:

* **80% training**
* **20% validation**

## Model Training

The classification models are trained using:

* **Loss:** Cross Entropy Loss
* **Optimizer:** AdamW
* **Epochs:** 5
* **Transfer Learning:** ImageNet pretrained weights

The best model weights based on validation accuracy are saved for later inference.

## Project Workflow

```text
Input Room Image
       ↓
Room Classification
(ViT / ResNet50 / VGG16)
       ↓
Predicted Room Type
       ↓
YOLOv8 Furniture Detection
       ↓
Furniture Cropping
       ↓
ResNet50 Feature Extraction
       ↓
2048-D Embeddings
       ↓
FAISS Similarity Search
       ↓
Similar Rooms / Furniture
       ↓
Amazon Search Links
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install ultralytics faiss-cpu kaggle scikit-learn
```

PyTorch and Torchvision are also required.

### 3. Configure Kaggle API

Download your `kaggle.json` API key from your Kaggle account and upload it when prompted by the notebook.

### 4. Open the notebook

Open:

```text
ai_based_interior_recommendation_system_.ipynb
```

You can run it using Google Colab or Jupyter Notebook.

### 5. Run the cells in order

The notebook will:

1. Install and import dependencies
2. Download the dataset
3. Preprocess the images
4. Train and compare classification models
5. Load YOLOv8
6. Detect furniture
7. Generate ResNet50 embeddings
8. Build FAISS indexes
9. Generate recommendations

## Example

The system accepts a room image and produces:

* Predicted room type
* Room classification confidence
* Detected furniture
* Similar room interiors
* Similar furniture
* E-commerce search links

## Project Structure

```text
AI-Interior-Design-Recommendation/
│
├── ai_based_interior_recommendation_system_.ipynb
├── README.md
├── yolov8m.pt
├── weights/
│   ├── resnet50.pt
│   ├── vgg16.pt
│   └── vit.pt
│
└── catalog/
    └── detected furniture crops
```

## Future Improvements

* Add EfficientNet-B3 to the model comparison
* Train YOLO on a dedicated furniture/interior dataset
* Replace generic Amazon search links with actual product-level recommendations
* Add a web or mobile interface
* Add user preference-based recommendations
* Add price, color, and style filtering
* Deploy the complete system as an API

## Author

**Jithin Kondeti**

B.Tech – Computer Science and Engineering (AI & ML)
