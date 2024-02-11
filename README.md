# Quarter 2 Project: Children Emotion Recognition

## Overview
This repository contains the codebase for the Children Emotion Recognition project. The project aims to develop a machine learning model capable of recognizing emotions from children's facial expressions. 

## Getting Started

### Prerequisites
- Conda (Anaconda or Miniconda)

### Environment Setup
To run the code, you need to set up a Python environment. The following instructions will guide you through creating and activating a Conda environment for this project.

1. **Create the Conda Environment**: Create a new Conda environment named `emotion_recognition` using Python 3.8 (replace `3.8` with the version used during development if different):

    ```bash
    conda create --name emotion_recognition python=3.8
    ```

2. **Activate the Environment**: Activate the newly created environment:

    ```bash
    conda activate emotion_recognition
    ```

3. **Install Dependencies**: Install the required packages specified in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
With the environment set up and activated, you can now run a select model by running the following:

```bash
cd scripts
python scripts/run_model_training.py
