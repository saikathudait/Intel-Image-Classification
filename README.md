#  Intel Image Classification using Deep Learning

A comprehensive deep learning project for classifying natural scene images into 6 categories using Custom CNN and Transfer Learning (EfficientNet).
# **Dataset Name: Intel Image Classification**
# Dataset Link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification?select=seg_train

##  Dataset

**Intel Image Classification Dataset**
- **Classes**: Buildings, Forest, Glacier, Mountain, Sea, Street
- **Training Images**: ~14,000
- **Testing Images**: ~3,000
- **Image Size**: 150x150 pixels

##  Project Highlights

-  Complete end-to-end deep learning pipeline
-  Extensive data visualization and EDA
-  Data augmentation for better generalization
-  Two model architectures:
  - Custom CNN (4 convolutional blocks)
  - Transfer Learning with EfficientNetB0
-  Comprehensive evaluation metrics
-  Confusion matrix analysis
-  Class-wise accuracy breakdown

##  Performance

======================================================================
CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

   buildings       0.90      0.89      0.90       437
      forest       0.98      0.98      0.98       474
     glacier       0.86      0.84      0.85       553
    mountain       0.86      0.85      0.85       525
         sea       0.89      0.90      0.90       510
      street       0.90      0.92      0.91       501

    accuracy                           0.90      3000
   macro avg       0.90      0.90      0.90      3000
weighted avg       0.90      0.90      0.90      3000


##  Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy, Pandas** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **OpenCV** - Image processing
- **Scikit-learn** - Metrics and evaluation

##  Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

##  Requirements.txt
```txt
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
plotly>=5.0.0
```

##  Usage

### 1. **Google Colab (Recommended)**

1. Click the "Open in Colab" badge above
2. Upload your dataset to Google Drive
3. Update the `base_path` variable in the notebook
4. Run all cells

### 2. **Local Environment**
```python
# Download the dataset
# Place seg_train.zip, seg_test.zip, seg_pred.zip in a folder

# Update paths in the notebook
base_path = '/path/to/your/dataset/'

# Run the notebook
jupyter notebook Intel_Image_Classification_Complete_Pipeline.ipynb
```

##  Model Architecture

### Custom CNN
- 4 Convolutional Blocks (32, 64, 128, 256 filters)
- Batch Normalization after each Conv layer
- MaxPooling and Dropout for regularization
- Dense layers: 512 → 256 → 6 (output)

### Transfer Learning
- Base: EfficientNetB0 (pre-trained on ImageNet)
- Global Average Pooling
- Dense layers: 512 → 256 → 6 (output)
- Dropout and Batch Normalization

##  Results

### Training History
![Training History](images/training_history.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Sample Predictions
![Predictions](images/sample_predictions.png)

##  Key Features

1. **Data Preprocessing**
   - Image resizing to 150x150
   - Normalization (pixel values 0-1)
   - Train-validation split (80-20)

2. **Data Augmentation**
   - Rotation (30°)
   - Width/Height shift (20%)
   - Shear and Zoom (20%)
   - Horizontal flip

3. **Training Strategy**
   - Early stopping (patience=10)
   - Learning rate reduction (factor=0.5)
   - Model checkpointing (best weights)

4. **Evaluation**
   - Accuracy and Loss metrics
   - Confusion matrix
   - Classification report
   - Class-wise accuracy

##  Dataset Structure
```
dataset/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── seg_test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_pred/
    └── (unlabeled images)
```


##  **.gitignore** (Copy this)
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Dataset files (too large for GitHub)
*.zip
seg_train/
seg_test/
seg_pred/
dataset/

# Model files (if > 100MB)
*.h5
*.hdf5
models/*.h5

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log