# Import necessary modules
import tensorflow as tf
from data_preparation import load_and_preprocess_data

def main():
    # Load and preprocess the MNIST dataset
    _, (x_test, y_test) = load_and_preprocess_data()

    # Load the trained model from the 'models' directory
    model = tf.keras.models.load_model('../models/epic_num_reader.keras')

    # Evaluate the model on the test data
    # - x_test: Test images
    # - y_test: Test labels
    val_loss, val_acc = model.evaluate(x_test, y_test)

    # Print the evaluation results
    # - val_loss: Loss value on the test set
    # - val_acc: Accuracy on the test set
    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)

if __name__ == '__main__':
    main()
