# Identifying Road Objects for Self-Driving Cars Using Convolutional Neural Networks

**Author:** Omkar Sreekanth  

## Overview
This project explores how convolutional neural networks (CNNs) can be used to identify common road objects in images for self-driving cars. The dataset included five classes—**cars, trucks, pedestrians, bicyclists, and lights**—with strong class imbalance (cars heavily overrepresented).

The work demonstrates preprocessing raw images, building multiple CNN models, and testing different regularization strategies to address overfitting. The best model achieved **83.32% validation accuracy**, showing proof of concept but also highlighting the need for further refinement for real-world deployment.


## Dataset
- **Source:** [Kaggle dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars?resource=downloadLinks)  
- **Classes & counts:**  
  - Cars: 101,312  
  - Trucks: 6,313  
  - Pedestrians: 10,637  
  - Bicyclists: 1,442  
  - Lights: 12,700  
- Bounding box annotations were used to crop individual objects from larger images.  
- Images resized to **32×32 pixels** for training.


## Methods

### Preprocessing
- Cropped sub-images of individual objects into class-specific folders.
- Split into training and validation sets.
- Applied class weighting to address dataset imbalance.

### CNN Architecture
- 4 CNN models, each with:  
  - **3 convolutional layers (3×3 kernels)**  
  - **2 max-pooling layers (2×2)**  
  - **1 flatten layer**  
  - **2 dense layers (512 ReLU, 5 softmax)**  
- Regularization techniques tested:  
  - Early stopping  
  - L2 (ridge) regularization with multiple lambda values  
  - Dropout (0.2–0.4 rates)


## Results
- **Unregularized CNN:** ~50% accuracy, severe overfitting.  
- **Early stopping:** ~81% accuracy, more stable training.  
- **Early stopping + L2:** **83.32% accuracy (best model)**, most stable validation performance.  
- **Early stopping + dropout:** ~77% accuracy, fluctuating performance.  


## Discussion
- All three regularization methods outperformed the baseline model.  
- Early stopping + L2 worked best, showing stable convergence and highest accuracy.  
- Class imbalance likely still limited performance—data augmentation could help.  
- At **83.32%**, accuracy is promising but insufficient for self-driving safety requirements.  


## Future Work
- Use larger, higher-quality images.  
- Augment minority classes (e.g., bicyclists, trucks).  
- Explore architectures for **multi-object detection** and bounding box prediction.  
- Experiment with different numbers of convolutional and dense layers.  


## Installation & Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Required Python packages:

* keras
* pandas
* numpy
* matplotlib
* scikit-learn

### Data Setup

1. Download dataset from Kaggle.
2. Unzip into an `images/` folder in the project directory.
3. Create the following folders for preprocessing:

```
train_data_cropped_new/
    ├── car/
    ├── truck/
    ├── pedestrian/
    ├── light/
    └── bicyclist/

val_data_cropped_new/
    ├── car/
    ├── truck/
    ├── pedestrian/
    ├── light/
    └── bicyclist/
```

4. Run the preprocessing steps in the Jupyter notebook.
5. Update variables `crop_data_train` and `crop_data_val` as needed.


## Usage

Open the Jupyter notebook:

```bash
jupyter notebook
```

Inside the notebook:

* Set `redo_model_X = True` to retrain a specific model.
* Run cells sequentially to preprocess, train, and evaluate models.

## Repository Structure

```
├── notebooks/          # Jupyter notebooks
├── src/                # (optional) Python scripts
├── docs/               # Documentation and reports
│   └── Project_Report.pdf
├── requirements.txt
└── README.md
```


## Key Takeaways

* CNNs are effective for feature extraction in road-object images.
* Regularization (especially **early stopping + L2**) significantly improves performance.
* Accuracy of \~83% demonstrates promise, but highlights the challenge of reaching near-100% reliability for safety-critical applications like autonomous driving.

