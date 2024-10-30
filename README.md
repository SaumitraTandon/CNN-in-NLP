# CNN for Text Classification in NLP

This repository provides an example of using Convolutional Neural Networks (CNNs) for Natural Language Processing (NLP) tasks, specifically text classification. The project includes a Jupyter Notebook that demonstrates building and training a CNN model on a provided dataset to classify text into different categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Notebook Details](#notebook-details)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results and Evaluation](#results-and-evaluation)
- [Contributing](#contributing)

## Project Overview

Convolutional Neural Networks (CNNs) are commonly applied to image processing tasks, but they can also be effective in NLP for tasks such as text classification. This project applies a CNN model to a text classification dataset using embeddings and convolutional layers to extract features from text data. 

The notebook explores the preprocessing, model building, training, and evaluation processes to classify text documents into predefined categories.

## Dataset

The dataset used for this project is located in the file `bbc_text_cls.csv`, containing labeled text documents. The columns in the dataset are as follows:
- `category`: the category label for each text document (e.g., 'business', 'entertainment', 'politics', 'sport', 'tech').
- `text`: the raw text data for each document.

### Dataset Source
The dataset appears to be from BBC and includes text data on various topics. Each row represents one text document along with its associated category.

### Dataset Statistics
For information on dataset statistics such as the number of entries per category, refer to the "Data Exploration" section in the notebook.

## Notebook Details

The notebook `CNN_in_NLP.ipynb` contains the full workflow for text classification using CNNs. Below is an outline of each section:

1. **Data Loading and Exploration**:
   - Loads the dataset and examines the distribution of categories.
   - Provides visualizations to understand data distribution.

2. **Text Preprocessing**:
   - Text tokenization and sequence padding.
   - Preparation of data for input into the CNN model.

3. **Model Architecture**:
   - CNN layers are applied to process embeddings and extract text features.
   - Dropout and dense layers are added to build a model for text classification.

4. **Model Training and Evaluation**:
   - The model is compiled and trained on the preprocessed text data.
   - Performance is evaluated using metrics like accuracy and confusion matrix visualization.

5. **Results Visualization**:
   - Displays training and validation accuracy/loss over epochs.
   - Shows classification metrics (e.g., precision, recall, F1-score).

## Installation and Setup

To run the notebook, ensure you have Python installed along with the necessary libraries. 

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required libraries (listed in `requirements.txt` or install directly as shown below)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install the packages directly:
   ```bash
   pip install numpy pandas tensorflow matplotlib seaborn scikit-learn
   ```

3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `CNN_in_NLP.ipynb` in Jupyter and follow along.

## Usage

To run the notebook and replicate the experiments:

1. Load and preprocess the dataset.
2. Train the CNN model as specified in the notebook.
3. Evaluate model performance on the test dataset.
4. Observe visualizations and model evaluation metrics.

### Example
An example of running the notebook's CNN model pipeline:
```python
# Sample code snippet to preprocess data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['text'])
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
train_padded = pad_sequences(train_sequences, maxlen=200)
```

Refer to the notebook for full implementation details.

## Results and Evaluation

The notebook includes performance metrics such as accuracy, precision, recall, and F1-score, as well as visualizations of the model's accuracy and loss over time. It also provides a confusion matrix to show classification effectiveness across different categories.

Example results:
- **Training Accuracy**: ~85%
- **Validation Accuracy**: ~80%
- **Confusion Matrix**: Provides insights into model misclassifications.

## Contributing

Contributions are welcome! Please follow the steps below if you would like to contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure any new code follows the project’s coding style and includes relevant tests.
