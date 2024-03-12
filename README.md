# Optimizing Facial Expression Recognition Tasks for Children Utilizing CNN Architectures

## Introduction

This project is designed to [Briefly describe the core objective of the project, such as solving a specific problem, demonstrating a technology, or providing a useful tool for certain tasks]. By leveraging cutting-edge technologies and methodologies, including [mention any significant technologies, frameworks, or methods used, e.g., machine learning models, data analysis techniques, etc.], this project aims to deliver [mention the key outcomes or benefits of the project].

## Prerequisites

Before you begin, ensure you meet the following requirements:
- Python 3.8 or newer
- Connecting to a GPU is preferred.

## Installation

### Setting Up the Environment

To get started, clone the repository and set up a virtual environment:

```bash
cd projectname
python3 -m venv venv_name
source venv_name/bin/activate
```
Next, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Running the Project

To run the project, follow these steps:

1. **Preprocessing:** First, run the preprocessing script to prepare your data.
```bash
./run_preprocessing.sh
```

2. **Execution:** Use the run.py script with the following arguments to train and evaluate the model:
```bash
python run.py --model_arch [MODEL_ARCH] --classifier [CLASSIFIER] --training_type [TRAINING_TYPE] --dataset_name [DATASET_NAME] [--augment_data]
```

- `--model_arch`: Specifies the architecture of the model you wish to use. Options include 'vit', 'resnet', 'efficientnet', and 'none'. The default value is 'none'. Note that for classifiers other than CNN, specifying the model architecture is required.
- `--classifier`: Required. Specifies the type of classifier to use. Options include 'NN' (Neural Network), 'SVM' (Support Vector Machine), 'RF' (Random Forest), 'CNN' (Convolutional Neural Network).
- `--training_type`: Specifies whether to use full or partial training for models. Options include 'full', 'partial', and 'none'. The default value is 'none'. This is required for pretrained model classifiers, except when using CNN.
- `--dataset_name`: Required. Specifies the name of the dataset to use. Options include 'fer' and 'dartmouth'.
- `--augment_data`: An optional flag that, when set, enables data augmentation for training.

Example Command for CNN Classifier:

To run the project using a CNN classifier on the Dartmouth dataset, you can use the following simplified command since `--model_arch` and `--training_type` are not required for CNN:

```bash
python run.py --classifier CNN --dataset_name dartmouth
```
This command initiates the training and evaluation process for a CNN classifier with the Dartmouth dataset. The script handles data preprocessing, model training, validation, and testing, outputting the results upon completion.

### Explanation of `run.py`
The `run.py` script orchestrates the model training and evaluation process. It accepts various arguments to customize the execution according to the desired model architecture, classifier, and dataset. Here's a breakdown of its main functionalities:

**Data Preparation:** Based on the specified dataset, the script preprocesses the data, applying any requested augmentations, and splits it into training, validation, and testing sets.
**Model and Classifier Configuration:** Depending on the arguments, the script configures the chosen model architecture and classifier for the task.
**Training and Validation:** The model is trained and validated against the prepared datasets, with progress and metrics reported according to the specified number of epochs.
**Evaluation:** Finally, the model is evaluated on the test set, providing insights into its performance.


