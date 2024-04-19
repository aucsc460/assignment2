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
import torch.optim as optim # to use the Optimizer class to optimize code
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
            context_words_before = words[max(0, i - window_size) : i] # getting the context words before the target word at i
            context_words_after = words[i + 1 : min(len(words), i + window_size + 1)] # getting the context words after the target word at i
            context = context_words_before + context_words_after
            target = words[i]
            training_data.append((context, target)) # appending the training sample to the training data
    return training_data

#end generate_training_data
import torch.nn.init as init
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
        for i in range(10):
            # convert X[i] and y[i] to one hot vectors (two lines)
            if X[i]:
                # transform contexts into one hot vectors of type int for embedding layer
                ith_context_vect = create_one_hot_vectors(X[i], vocabulary).int()

                # transform labels into one hot vectors of type int for embedding layer
                y_label = one_hot_encode(y[i], vocabulary)

                # get expected predictions
                prediction = model(ith_context_vect)

                # apply prob softmax
                prob_softmax = F.softmax(prediction, dim=0)  # drop batch size dimension to compare loss

                # Compute cross entropy loss
                loss = F.cross_entropy(prediction, y_label)
                total_loss += loss

        # Backward step:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        list_total_loss.append(total_loss)

        list_epochs.append(epoch)

    #plot_graph(list_epochs, list_total_loss)
    torch.save(model.state_dict(), 'final_model_weights.pth')



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
    tensor = torch.zeros(1, len(vocab)) # Pytorch assumes everything is in batches, so we set batch size = 1
    tensor[0][index] = 1
    return tensor

def create_one_hot_vectors(input, vocab):
    """
    Converts a single training example into one hot vectors.

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
        # if (len(input) == 0):
        #     print ("======aha, WE'EVE FOUND HER!!!! =========")
    context_tensor = torch.stack(context_vector)
    return context_tensor

# ====================== VISUALIZING THE DATA ================== #
# REFERENCE: https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/

## NEED TO GET THE WORD_EMBEDDINGS FROM THE MODEL
## SAVE THE WEIGHTS AND LOAD THE WEIGHTS

def plot_PCA(word_embeddings):

    """
    Plots and visualizes the similarities in words from our trained model

    Uses my_PCA to perform PCA on the 2D word embeddings
    
    """

    # Perform PCA on the 2D word embeddings
    word_embeddings_reduced = my_PCA(word_embeddings)

    #All this plotting was referenced from geeksforgeeks.org
    plt.figure(figsize=(100, 100))
    plt.scatter(word_embeddings_reduced[:, 0], word_embeddings_reduced[:, 1], alpha=0.5)

    for i, word in enumerate(vocab_list.keys()):
        plt.annotate(word, xy=(word_embeddings_reduced[i, 0], word_embeddings_reduced[i, 1]), fontsize=8)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Word Embeddings')
    plt.grid(True)
    plt.show()

def my_PCA(word_embeddings):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    word_embeddings_reduced = pca.fit_transform(word_embeddings) # Transform our word embeddings to 2D
    return word_embeddings_reduced

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
# print('Processed text first line: ', processed_text[0].split())
# print('first six examples of training data: ', training_data[:6]) 


# Splitting training and testing data using the hold-out method (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(type(X_train))
# print(type(y_train))

# print('X_train first three examples: ', X_train[:3])
# print('y_train first three examples: ', y_train[:3])

context_vector = create_one_hot_vectors(X_train[:1][0], vocab_list)

CBOW_model = CBOW(vocab_size=len(vocab_list), hidden_size=100)

train(CBOW_model, vocab_list, X_train, y_train)

# Load the saved weights (parameters) of the trained CBOW model from a file
CBOW_model.load_state_dict(torch.load('final_model_weights.pth'))

# Extract the word embeddings from the CBOW model and convert them to a NumPy array
weights = CBOW_model.embedding.weight.detach().numpy()

# Plot the PCA visualization of the word embeddings
plot_PCA(weights)


# print(context_vector)
# print(context_vector[0])

# Creating the CBOW model using the CBOW class
# cbow = CBOW(vocab=vocab_list)








