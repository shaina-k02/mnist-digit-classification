# MNIST Digit Classification

This project demonstrates the classification of handwritten digits from the MNIST dataset using a neural network built with TensorFlow and Keras. Components include data preprocessing, model building, training, evaluation, and prediction components, organized in a clear and structured manner.
## Features

- **Data Preprocessing**: Load and normalize the MNIST dataset.
- **Model Building**: Define a neural network model using TensorFlow and Keras.
- **Training**: Train the model on the MNIST training data.
- **Evaluation**: Evaluate the model's performance on the test data.
- **Prediction**: Make predictions on new data using the trained model.
- **Model Saving and Loading**: Save the trained model and reload it for future use.
- **Visualization**: Display sample images and their predicted labels.

## Repository Structure

- `data/`: Contains datasets and data-related files.
- `notebooks/`: Jupyter notebooks for exploration and experimentation.
- `src/`: Source code for data preparation, model definition, training, and evaluation.
- `models/`: Directory to save trained models.
- `scripts/`: Shell scripts to run the project pipeline.
- `requirements.txt`: Project dependencies.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Jupyter (for notebooks)

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/mnist-digit-classification.git
   cd mnist-digit-classification
   ```

2. **Create a virtual environment and activate it** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the training script**:
   ```sh
   python src/train.py
   ```

2. **Evaluate the model**:
   ```sh
   python src/evaluate.py
   ```

3. **Run all scripts using the shell script**:
   ```sh
   sh scripts/run_all.sh
   ```

## Project Components

### Data Preprocessing

The data preprocessing script (`src/data_preparation.py`) handles loading and normalizing the MNIST dataset.

### Model Building

The model definition script (`src/model.py`) defines the neural network architecture and compiles the model.

### Training

The training script (`src/train.py`) trains the model on the MNIST training data and saves the trained model to the `models/` directory.

### Evaluation

The evaluation script (`src/evaluate.py`) loads the trained model and evaluates its performance on the test data.

### Jupyter Notebooks

The `notebooks/mnist_classification.ipynb` notebook provides an interactive way to explore and classify handwritten digits using a neural network built with TensorFlow and Keras.
