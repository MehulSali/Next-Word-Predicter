# Next Word Prediction Model

This project trains an LSTM-based neural network to predict the next word in a given sequence of text. The model uses an embedding layer, an LSTM layer, and a dense output layer with softmax activation.

## Features
- Uses **LSTM** for sequence prediction.
- Trained on a text dataset with **word embeddings**.
- Saves both the trained model and tokenizer for future inference.

## Installation
1. Clone the repository (if applicable).
2. Install dependencies.

## Training the Model
Train the model using the provided script.

## Using the Model for Prediction
Load the model and tokenizer, then use them for next-word prediction.

## Notes
- Ensure that the dataset is properly preprocessed before training.
- The tokenizer should be used consistently for tokenizing input text both during training and inference.
- Adjust the number of epochs and batch size based on dataset size and available computational resources.

## License
This project is open-source under the MIT License.
