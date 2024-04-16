"""
File: cbow.py

Author(s): Anjola Aina, Aarushi Gupta, Priscilla Adebanji, James Rota, Cindy Ni
Date: April 15th, 2024

Description:

This file contains all the necessary functions and class used to implement the continous bag of words (CBOW) algorithm.

It contains the following methods to process the data and generate the training data:
    - process_data(data) -> text, vocabulary
    - generate_training_data(text, window_size) -> training_data
        - NOTE: The window size has a default value of two, so the only required parameter is the text param.

The following class is used to implement the CBOW model:
    - CBOW(vocab_size, hidden_size, embedding_dim)
        - forward(x) -> probability distribution over vocab, with highest value giving the prediction for the given input x

The following functions are used to train the model:
    - train(model, X, y, eopchs, lr, weight_decay) -> void
    - NOTE: L2 normalization was added to the train function via the weight_decay parameter.

The following functions are used to visualize the model:
    - plot_graph(list_epochs, list_total_loss) -> void
    - my_PCA(word_embedding) -> Any
    - PCA(word_embedding) -> void

Sources:
    - To generate regex expressions: https://regex101.com/
    - Special characters: https://saturncloud.io/blog/how-to-remove-special-characters-in-pandas-dataframe/#:~:text=Use%20regular%20expressions&text=In%20this%20method%20with%20regular,character%2C%20effectively%20removing%20special%20characters.
    - Tokenize each sentence: https://medium.com/@saivenkat_/implementing-countvectorizer-from-scratch-in-python-exclusive-d6d8063ace22
    - CountVectorizer: https://spotintelligence.com/2022/12/20/bag-of-words-python/#:~:text=Scikit%2DLearn-,In%20Python%2C%20you%20can%20implement%20a%20bag%2Dof%2Dwords,CountVectorizer%20class%20in%20the%20sklearn.
    - Idea for one hot encode function: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    - Visualizing using PCA: https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/ 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
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
    # Removes special characters from the 'Line' column
    data = data.replace(special_char, '', regex=True)
    # Lowercases all letters within the 'Line' column 
    data = data.str.lower()  
    text = data.values

    # Regex expression that ensures that we only get single characters or words, NO numbers
    token_pattern = r'[a-zA-Z]+|[a-zA-Z]'

    # Creates the vocabulary and tokenzies the text
    vector = CountVectorizer(token_pattern=token_pattern)
    vector.fit(text)

    return text, vector.vocabulary_

# ====================== GENERATING TRAINING DATA FOR MODEL ======================

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
        words = sentence.split()  # Splits words in processed text
        for i in range(len(words)):
            # Getting the context words before the target word at i
            context_words_before = words[max(0, i - window_size): i]  
            # Getting the context words after the target word at i
            context_words_after = words[i + 1: min(len(words),
                                                   i + window_size + 1)]  
            context = context_words_before + context_words_after
            target = words[i]
            # Appending the training sample to the training data
            training_data.append((context, target)) 
    return training_data

# ====================== CBOW MODEL ======================

class CBOW(nn.Module):
    """
    This class implements the CBOW model. It inherits all attributes from its base class, the Module class.
    It creates the embedding and MLP layers, along with the ReLU activiation function.
    """
    def __init__(self, vocab_size, hidden_size, embedding_dim=100):
        super(CBOW, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(embedding_dim, hidden_size) # Linear = fully connected layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, vocab_size) # Linear = fully connected layer

    # Prediction Function
    def forward(self, x):
        """
        Implements the forward pass for the CBOW architecture.

        Args:
            x (Tensor): the input to the CBOW model.

        Returns:
            Any: the log probability of the model (i.e., the prediction).
        """
        # 4D result vector (batch_size, seq_len, vocab_size, dim_size)
        embeddings = self.embedding(x)  
        # Geting average embeddings across vocabulary
        average_embeddings = torch.mean(embeddings, dim=2)  
        hidden_output = self.relu(self.hidden(average_embeddings))
        output = self.output(hidden_output)

        return output[0]

# ====================== TRAINING THE MODEL ======================

# TRAINING FUNCTION, PASS THE CBOW MODEL INTO IT AND USE IT HERE
def train(model, vocabulary, X, y, epochs=1000, lr=0.01, weight_decay=0.01):
    """
    Trains the model.

    Args:
        model (Any): The CBOW model.
        X (Tensor): The input values to the model.
        y (Tensor): The corresponding values for the model.
        epochs (int, optional): The specified number of iterations to go through the training data. Defaults to 1000.
        lr (float, optional): The learning rate to be applied to the SGD. Defaults to 0.01.
        weight_decay(float, optional): L2 normalization. Defaults to 0.01.
    """
    list_total_loss = []
    list_epochs = []
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        # Reset total loss for each iteration through training set
        total_loss = 0

        # Iterate through training data X
        for i in range(len(X)):
            if X[i]: # some training examples are empty -> []
                
                # Transform contexts into one hot vectors of type int for embedding layer
                ith_context_vect = create_one_hot_vectors(X[i], vocabulary).int()

                # Transform labels into one hot vectors of type int for embedding layer
                y_label = one_hot_encode(y[i], vocabulary)

                # Get expected predictions
                prediction = model(ith_context_vect)

                # Compute cross entropy loss
                loss = F.cross_entropy(prediction, y_label)
                total_loss += loss

                # Backward step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # To plot the loss as a function of the epochs (visualizing the loss)
        list_total_loss.append(total_loss.detach().numpy())
        list_epochs.append(epoch)

    plot_graph(list_epochs, list_total_loss)
    torch.save(model.state_dict(), 'final_model_weights.pth')

def plot_graph(list_epochs: list, list_total_loss: list):
    """
    This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

    Args:
        list_epochs (list): The list of epochs (iterations).
        list_total_loss (list): The list of total losses per epoch.
    """
    print(type(list_epochs))
    print(type(list_total_loss))
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

# ====================== VISUALIZING THE DATA ====================== 

def plot_PCA(word_embeddings):

    """
    Plots and visualizes the similarities in words from the trained model. 
    Uses my_PCA to perform PCA on the 2D word embeddings.
    
    Args:
        word_embeddings (Any): The word embeddings from the model.
    
    """
    # Perform PCA on the 2D word embeddings
    word_embeddings_reduced = my_PCA(word_embeddings)

    # All this plotting was referenced from geeksforgeeks.org
    plt.figure(figsize=(8, 8))
    plt.scatter(word_embeddings_reduced[:, 0], word_embeddings_reduced[:, 1], alpha=0.5)

    for i, word in enumerate(vocab_list.keys()):
        plt.annotate(word, xy=(word_embeddings_reduced[i, 0], word_embeddings_reduced[i, 1]), fontsize=8)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Word Embeddings')
    plt.grid(True)
    plt.show()

def my_PCA(word_embeddings):
    """
    Reduces the dimension of the word embeddings to 2.

    Args:
        word_embeddings (Any):  The word embeddings from the model.

    Returns:
        Any: The word embeddings reduced to 2 dimensions for visualization.
    """
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    word_embeddings_reduced = pca.fit_transform(word_embeddings) # Transform our word embeddings to 2D
    return word_embeddings_reduced

# ====================== TESTING THE MODEL ======================

# Reading file and processing the data
df = pd.read_csv('shakespeare.txt', sep='\t', header=None, names=['Line'])
processed_text, vocab_list = process_data(df['Line'])

# Generating the training data
training_data = generate_training_data(text=processed_text)

# Splitting the training data into X and y pairs
X = [data[0] for data in training_data]
y = [data[1] for data in training_data]

# Splitting training and testing data using the hold-out method (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=42)

print('Length of X_train', len(X_train))

# Getting the vocabulary size
vocab_len = len(vocab_list)

# Initializing the model
CBOW_model = CBOW(vocab_size=vocab_len, hidden_size=1)

# Training the model
train(model=CBOW_model, vocabulary=vocab_list, X=X_train, y=y_train)

# Loading the saved weights (parameters) of the trained CBOW model from a file
CBOW_model.load_state_dict(torch.load('final_model_weights.pth'))

# Extracting the word embeddings from the CBOW model and convert them to a NumPy array
weights = CBOW_model.embedding.weight.detach().numpy()

# Plotting the PCA visualization of the word embeddings
plot_PCA(weights)