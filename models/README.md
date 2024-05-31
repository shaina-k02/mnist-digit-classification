# Models Directory

This directory is used to store the trained models for the MNIST digit classification project. The models are saved in a format that can be easily loaded and used for evaluation and predictions.

## Contents

- `epic_num_reader.keras`: This file contains the trained model saved after the training process. The model is saved in the Keras native format (`.keras`), which includes the model architecture, weights, and training configuration.

## Usage

### Saving a Model

In the training script (`src/train.py`), the model is saved using the following command:

```python
model.save('../models/epic_num_reader.keras')
```

This command saves the entire model (architecture, weights, and training configuration) to the specified file.

### Loading a Model

To load a saved model for evaluation or making predictions, use the following command in your script:

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('models/epic_num_reader.keras')
```

This command loads the model from the specified file, making it ready for evaluation or prediction.

### Evaluating a Loaded Model

After loading the model, you can evaluate it on the test data to check its performance:

```python
# Evaluate the model on the test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: ", val_loss)
print("Accuracy: ", val_acc)
```

### Making Predictions with a Loaded Model

You can also use the loaded model to make predictions on new data:

```python
# Make predictions on the test data
predictions = model.predict(x_test)

# Print the true label and the predicted label for a specific test image
print('True Label: ', y_test[2])
print('Predicted Label: ', np.argmax(predictions[2]))
```

## Additional Information

### Model Architecture

The saved model includes the following architecture:

- **Conv2D Layer**: 32 filters, kernel size of 3x3, ReLU activation, input shape of (28, 28, 1)
- **MaxPooling2D Layer**: Pool size of 2x2
- **Conv2D Layer**: 64 filters, kernel size of 3x3, ReLU activation
- **MaxPooling2D Layer**: Pool size of 2x2
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU activation
- **Output Dense Layer**: 10 units, softmax activation

### Training Configuration

The model is compiled with the following configuration:

- **Optimizer**: Adam
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Accuracy

### Benefits of Using the Keras Native Format

- **Complete Model Saving**: Saves the architecture, weights, and training configuration.
- **Ease of Use**: Can be easily loaded with a single command for further training, evaluation, or predictions.
- **Compatibility**: Ensures compatibility with TensorFlow and Keras environments.

This directory structure and usage guide help keep the project organized and make it easy to manage and utilize trained models.
```
