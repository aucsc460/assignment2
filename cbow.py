"""
File: cbow.py

Author(s): Anjola Aina, Aarushi Gupta, Priscilla Adebanji, James Rota, Cindy Ni
Date: April 13th, 2024

Description: 

This file contains all the necessary functions and class used to implement the continous bag of words (CBOW) algorithm.

It contains the following methods to process the data and generate the training data:
    - process_data(data) -> text, vocabulary
    - generate_training_data(text, window_size) -> training_data
        - NOTE: The window size has a default value of two, so the only required parameter is the text param.
        
The following class is used to implement the CBOW model:
    - CBOW(vocab_size, hidden_size, embedding_dim)
        - forward(x) -> probability distribution over vocab, with highest value giving the prediction for the given input x
        
The following functions are used to train the model (and apply techniques to prevent overfitting):
    - train(model, X, y) -> void
    - plot_graph(list_epochs, list_total_loss) -> void
    - TODO: early_stopping / regularization? - this function would be used in conjuction with the train i assume
    
The following functions are used to visualize the model:
    - TODO: insert function definition here

Sources:
    - Special characters: https://saturncloud.io/blog/how-to-remove-special-characters-in-pandas-dataframe/#:~:text=Use%20regular%20expressions&text=In%20this%20method%20with%20regular,character%2C%20effectively%20removing%20special%20characters.
    - Tokenize each sentence: https://medium.com/@saivenkat_/implementing-countvectorizer-from-scratch-in-python-exclusive-d6d8063ace22
    - CountVectorizer: https://spotintelligence.com/2022/12/20/bag-of-words-python/#:~:text=Scikit%2DLearn-,In%20Python%2C%20you%20can%20implement%20a%20bag%2Dof%2Dwords,CountVectorizer%20class%20in%20the%20sklearn. 
    - Idea for one hot encode function: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim # to use the Optimizer class to optimize code
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# ====================== PREPROCESSING THE DATA ======================

def process_data(data):
    """
    Processes the data by removing all special characters, converting all words to lowercase, tokenizing all the sentences, and
    creating the vocabulary. The function returns the text that has been preprocessed and the vocabulary.

    Args:
        data (Any): The data containing the text to be preprocessed.

    Returns:
        tuple(ArrayLike, dict): Returns the processed text and the vocabulary.
    """
    # All special characters are defined using regex (regular expressions)
    special_char = r'[^\w\s]'
    data = data.replace(special_char, '', regex=True) # removes special characters from the 'Line' column
    data = data.str.lower() # lowercases all letters within the 'Line' column

    # Tokenizes each sentence
    text = data.values
    vocab = set()
    for i in text:
        for j in i.split(' '):
            if len(j) > 2:
                vocab.add(j)

    # Create the vocabulary
    vector = CountVectorizer()
    vector.fit(vocab)
    vector.transform(vocab)

    return text, vector.vocabulary_

# ====================== GENERATING TRAINING DATA FOR MODEL ======================

# NOTE: I CHANGED THIS CODE SO I COULD WORK ON CONSTRUCTING THE ARCHITECTURE FOR CBOW
def generate_training_data(text, window_size=2):
    """
    Generates the training data for the CBOW algorithm.

    Args:
        text (ArrayLike): The processed text.
        window_size (int, optional): The context window size. Defaults to 2.

    Returns:
        training_data (tuple): the training data for the algorithm, where each training example is tuple (context, target), where context refers to the context words before and after the target word, and target refers to thw word in between them (the word we want to predict).
    """
    training_data = []
    for sentence in text:
        words = sentence.split()  # split words in processed text
        for i in range(len(words)):
            context_words_before = words[max(0, i - window_size) : i] # getting the context words before the target word at i
            context_words_after = words[i + 1 : min(len(words), i + window_size + 1)] # getting the context words after the target word at i
            context = context_words_before + context_words_after
            target = words[i]
            training_data.append((context, target)) # appending the training sample to the training data
    return training_data

# ====================== CBOW MODEL ======================

class CBOW(nn.Module):
    """
    This class implements the CBOW model. It inherits all attributes from its base class, the Module class.
    It creates the embedding and MLP layers, along with the ReLU and LogSoftmax activiation functions.
    """
    def __init__(self, vocab_size, hidden_size, embedding_dim=100):
        super(CBOW, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_size) # Linear = fully connected layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=hidden_size, out_features=vocab_size) # Linear = fully connected layer
        
        # Softmax Layer
        self.log_softmax = nn.LogSoftmax() # chose log softmax to avoid problems with multiplying probabilites and getting values too small

    # Prediction Function
    def forward(self, x):
        """
        Implements the forward pass for the CBOW architecture.

        Args:
            x (Tensor): the input to the CBOW model.

        Returns:
            Any: the log probability of the model (i.e., the prediction).
        """
        embeddings = self.embedding(x)
        average_embeddings = torch.mean(embeddings) # gonna test it out later
        hidden_output = self.relu(self.hidden(average_embeddings))
        output = self.output(hidden_output)
        prob = self.log_softmax(output)
        return prob
    
# ====================== TRAINING THE MODEL ======================

# TRAINING FUNCTION, PASS THE CBOW MODEL INTO IT AND USE IT HERE
def train(model, X, y):
    """
    Trains the model.

    Args:
        model (Any): the CBOW model.
        X (Tensor): the input values to the model.
        y (Tensor): the corresponding values for the model.
    """
    total_loss = 0
    list_total_loss = []
    list_epochs = []
    
    
    # AFTER TRAINING THE DATA, CALL THIS FUNCTION FOR VISUALIZATION
    plot_graph(list_epochs, list_total_loss)

def plot_graph(self, list_epochs: list, list_total_loss: list):
    """
    This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

    Args:
        list_epochs (list): list of epochs (iterations)
        list_total_loss (list): list of total losses per epoch
    """
    fig, ax = plt.subplots()
    ax.plot(list_epochs, list_total_loss) 
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Total loss')
    ax.set_title('Loss Function as a Function of Epochs')
    plt.show()
    
def word_to_index(word: str, vocab: dict):
    """
    Gets the index of the word in the vocabulary dictionary.
    
    Args:
        word (str): The word (key) to retrieve its corresponding index in the vocabulary.
        vocab (dict): The vocabulary, consisting of all unique words in the document.
        
    Returns:
        int: The corresponding index (value) of the word (key).
    """
    return vocab.get('word')

def one_hot_encode(word, vocab: dict):
    """
    Turns a word into a one hot vector.
    
    Args:
        word (str): The word to be turned into a one hot vector.
        vocab (dict): The vocabulary, consisting of all unique words in the document.
        
    Returns:
        tensor: The one hot vector representation of the word.
    """
    index = word_to_index(word, vocab)
    tensor = torch.zeros(1, len(vocab)) # Pytorch assumes everything is in batches, so we set batch size = 1
    tensor[0][index] = 1
    return tensor
    
# ====================== TESTING THE MODEL ======================

# Reading file and creating pandas dataframe
df = pd.read_csv('shakespeare.txt', sep='\t', header=None, names=['Line'])
processed_text, vocab_list = process_data(df['Line'])

# Generating the training data
training_data = generate_training_data(text=processed_text)

# Splitting the training data into X and y pairs
X = [data[0] for data in training_data]
y = [data[1] for data in training_data]

# NOTE: TESTING TO ENSURE THAT DATA ACTUALLY WORKS - will be deleted later
print('Processed text first line: ', processed_text[0].split())
print('first six examples of training data: ', training_data[:6]) 


# Splitting training and testing data using the hold-out method (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(type(X_train))
print(type(y_train))

print('X_train first three examples: ', X_train[:3])
print('y_train first three examples: ', y_train[:3])

print(vocab_list.get('creatures'))
print(one_hot_encode(X_train[0][0], vocab_list))

# X_train_1 = one_hot_encode()

# Creating the CBOW model using the CBOW class
# cbow = CBOW(vocab_size=len(vocab_list), hidden_size=128)
# train(cbow, X_train, y_train)

