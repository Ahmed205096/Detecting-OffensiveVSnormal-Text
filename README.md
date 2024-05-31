# Detecting-OffensiveVSnormal-Text

Dataset: jigsaw-toxic-comment-classification-challenge

This repository contains code for building a model that detects offensive language in text. It leverages a Bidirectional LSTM architecture to classify text into different categories of offensive content.

**Getting Started**

### Prerequisites

- Python 3.x (with TensorFlow 2.x or later)
- pandas
- Keras

You can install these dependencies using a package manager like `pip`:

```bash
pip install tensorflow pandas keras
```

**Data**

This code assumes you have access to the Jigsaw Toxic Comment Classification Challenge dataset ([https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)). You'll need to place the `train.csv` file from the dataset in the `jigsaw-toxic-comment-classification-challenge` directory within your project.

**Running the Script**

1. Clone this repository.
2. Navigate to the project directory in your terminal.
3. Run the script:

```bash
python main.py
```

**Code Breakdown**

The script follows these steps:

1. **Install Dependencies and Load Data:**
   - Imports necessary libraries (TensorFlow, pandas, Keras).
   - Reads the `train.csv` file and preprocesses the data.

2. **Preprocess:**
   - Renames the `comment_text` column to `tweet`.
   - Splits the data into features (`tweet`) and labels (`y`).
   - Creates a `TextVectorization` layer to convert text into numerical sequences.
   - Prepares the dataset for training, validation, and testing.

3. **Create Sequential Model:**
   - Defines a sequential model using Keras.
   - Embeds text into numerical vectors.
   - Uses a Bidirectional LSTM layer to capture long-range dependencies in sequences.
   - Adds fully connected layers for feature extraction.
   - Applies a sigmoid activation function in the output layer for multi-label classification.

4. **Compile and Train the Model:**
   - Compiles the model with a binary cross-entropy loss function and the Adam optimizer.
   - Trains the model for a specified number of epochs.

5. **Evaluate Model (**This section is missing from the provided code snippet but can be added**):**
   - Defines metrics (precision, recall, accuracy) to evaluate the model's performance.
   - Iterates through the test data, makes predictions, and updates evaluation metrics.

**Note:**

- The provided code snippet only shows the training phase. You'll need to implement the evaluation section (`In[ ]`) to calculate performance metrics on the held-out test set.

**Further Enhancements**

- Experiment with different hyperparameters (learning rate, number of epochs, etc.) to improve model performance.
- Explore other text pre-processing techniques (e.g., stemming, lemmatization).
- Consider using a pre-trained word embedding model (e.g., Word2Vec, GloVe) to improve the embedding layer.

This is a basic framework for detecting offensive language. You can extend it to fit your specific needs and incorporate more advanced techniques.
