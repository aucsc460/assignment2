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
    - To generate regex expressions: https://regex101.com/
    - Special characters: https://saturncloud.io/blog/how-to-remove-special-characters-in-pandas-dataframe/#:~:text=Use%20regular%20expressions&text=In%20this%20method%20with%20regular,character%2C%20effectively%20removing%20special%20characters.
    - Tokenize each sentence: https://medium.com/@saivenkat_/implementing-countvectorizer-from-scratch-in-python-exclusive-d6d8063ace22
    - CountVectorizer: https://spotintelligence.com/2022/12/20/bag-of-words-python/#:~:text=Scikit%2DLearn-,In%20Python%2C%20you%20can%20implement%20a%20bag%2Dof%2Dwords,CountVectorizer%20class%20in%20the%20sklearn.
    - Idea for one hot encode function: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # to use the Optimizer class to optimize code
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
    data = data.replace(special_char, '', regex=True)  # removes special characters from the 'Line' column
    data = data.str.lower()  # lowercases all letters within the 'Line' column
    text = data.values

    # Regex expression that ensures that we only get single characters or words, NO numbers
    token_pattern = r'[a-zA-Z]+|[a-zA-Z]'

    # Creates the vocabulary and tokenzies the text
    vector = CountVectorizer(token_pattern=token_pattern)
    vector.fit(text)

    # TESTING - DELETE LATER
    """ print(vector.vocabulary_.get('we'))
    print(vector.vocabulary_.get('i'))
    print(vector.vocabulary_.get('1'))
    print(vector.vocabulary_.get('iv'))
    print(vector.vocabulary_.get('act')) """

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
            context_words_before = words[max(0, i - window_size): i]  # getting the context words before the target
            # word at i
            context_words_after = words[i + 1: min(len(words),
                                                   i + window_size + 1)]  # getting the context words after the target word at i
            context = context_words_before + context_words_after
            target = words[i]
            training_data.append((context, target))  # appending the training sample to the training data
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
        self.hidden = nn.Linear(embedding_dim, hidden_size)  # Linear = fully connected layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, vocab_size)  # Linear = fully connected layer

    # Prediction Function
    def forward(self, x):
        """
        Implements the forward pass for the CBOW architecture.

        Args:
            x (Tensor): the input to the CBOW model.

        Returns:
            Any: the log probability of the model (i.e., the prediction).
        """
        embeddings = self.embedding(x)  # 4D result vector (batch_size, seq_len, vocab_size, dim_size)
        # print("embeddings shape: ", embeddings.shape)

        average_embeddings = torch.mean(embeddings, dim=2)  # get average embeddings across vocabulary
        # print("average shape: ", average_embeddings.shape)

        hidden_output = self.relu(self.hidden(average_embeddings))
        # print("hidden shape: ", hidden_output.shape)

        output = self.output(hidden_output)
        # print("output pre-softmax shape: ", output.shape)

        # # Apply Softmax
        # prob = F.log_softmax(output, dim=2)  # get softmax probabilities across vocabulary
        # # print("probability shape: ", prob.shape)
        #
        # predicted_value = prob[0]

        return output[0]


# ====================== TRAINING THE MODEL ======================

# TRAINING FUNCTION, PASS THE CBOW MODEL INTO IT AND USE IT HERE
def train(model, vocabulary, X, y, epochs=5, lr=0.001):
    """
    Trains the model.

    Args:
        model (Any): the CBOW model.
        X (Tensor): the input values to the model.
        y (Tensor): the corresponding values for the model.
        --------------
        :param y:
        :param X:
        :param model:
        :param vocabulary:
        :param lr:
        :param epochs:
    """
    list_total_loss = []
    list_epochs = []
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # reset total loss for each iteration through training set
        total_loss = 0

        # iterate through training data X
        for i in range(len(X)):
            # convert X[i] and y[i] to one hot vectors (two lines)
            if X[i]:
                # transform contexts into one hot vectors of type int for embedding layer
                ith_context_vect = create_one_hot_vectors(X[i], vocabulary).int()

                # transform labels into one hot vectors of type int for embedding layer
                y_label = one_hot_encode(y[i], vocabulary)

                # get expected predictions
                prediction = model(ith_context_vect)

                # Compute cross entropy loss
                loss = F.cross_entropy(prediction, y_label)
                total_loss += loss

                # Backward step:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        list_total_loss.append(total_loss)

        list_epochs.append(epoch)

    for i in range(len(list_total_loss)):
        print(list_total_loss[i].item())

    #plot_graph(list_epochs, list_total_loss)
    torch.save(model.state_dict(), 'final_model_weights.pth')



def plot_graph(self, list_epochs: list, list_total_loss: list):
    """
    This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

    Args:
        self: a model object
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
    return vocab.get(word)


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
    tensor = torch.zeros(1, len(vocab))  # Pytorch assumes everything is in batches, so we set batch size = 1
    tensor[0][index] = 1
    return tensor


def create_one_hot_vectors(input, vocab):
    """
    Converts a single training example (i.e. a context group) into one hot vectors.

    Args:
        input (list): The training example to be converted into one hot vectors.
        vocab (dict): The vocabulary, consisting of all unique words in the document.

    Returns:
        tensor: A tensor containing all one hot vector representations of the input.
    """
    context_vector = []
    for i in range(len(input)):
        one_hot = one_hot_encode(input[i], vocab)
        context_vector.append(one_hot)
    context_tensor = torch.stack(context_vector)
    return context_tensor


# ====================== TESTING THE MODEL ======================

# Reading file and creating pandas dataframe
df = pd.read_csv('shakespeare.txt', sep='\t', header=None, names=['Line'])
processed_text, vocab_list = process_data(df['Line'])

# Generating the training data
training_data = generate_training_data(text=processed_text)

# Splitting the training data into X and y pairs
X = [data[0] for data in training_data]
y = [data[1] for data in training_data]

# Splitting training and testing data using the hold-out method (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get vocabulary size
vocab_len = len(vocab_list)

# Initialise model
CBOW_model = CBOW(vocab_size=vocab_len, hidden_size=1)

# Train
train(model=CBOW_model, vocabulary=vocab_list, X=X_train, y=y_train)

# print(type(X_train))
# print(type(y_train))
#
# print('X_train first three contexts: ', X_train[2])
# print('y_train first three labels: ', y_train[2])
#
# # TRAINING
# context_vector = create_one_hot_vectors(X_train[2], vocab_list)
#
# print(len(X_train))
# print(len(vocab_list))
# print(context_vector.type)
# print(context_vector.shape)

# To do:
# Figure out the expected shape of the layers
# Ensure the output is expected


# Creating the CBOW model using the CBOW class
# cbow = CBOW(vocab_size=len(vocab_list), hidden_size=128)
# train(cbow, X_train, y_train)

"""
===============
Changes made
===============

* Converted One-Hot Vectors to Ints before embedding

why: 
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: 
Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)


* Added dimension to average_embeddings
why: needed to specify to calculate the average with respect to the vocabulary size. 

"""
