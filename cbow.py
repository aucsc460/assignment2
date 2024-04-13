# IMPORTS
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim # to use the Optimizer class to optimize code
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Functions used for preprocessing data and generating training data

def process_data(data):
    # Source for special characters: https://saturncloud.io/blog/how-to-remove-special-characters-in-pandas-dataframe/#:~:text=Use%20regular%20expressions&text=In%20this%20method%20with%20regular,character%2C%20effectively%20removing%20special%20characters.
    # all special characters are defined using regex (regular expressions)
    special_char = r'[^\w\s]'
    # Remove special characters from the 'Line' column
    data = data.replace(special_char, '', regex=True)

    # Lowercase all letters within the 'Line' column
    data = data.str.lower()

    # Tokenize each sentence
    # https://medium.com/@saivenkat_/implementing-countvectorizer-from-scratch-in-python-exclusive-d6d8063ace22
    text = data.values
    vocab = set()
    for i in text:
        for j in i.split(' '):
            if len(j) > 2:
                vocab.add(j)

    # create vocabulary
    vector = CountVectorizer()
    vector.fit(vocab)
    vector.transform(vocab)

    # https://spotintelligence.com/2022/12/20/bag-of-words-python/#:~:text=Scikit%2DLearn-,In%20Python%2C%20you%20can%20implement%20a%20bag%2Dof%2Dwords,CountVectorizer%20class%20in%20the%20sklearn.
    # Print vocabulary
    # print(vector.vocabulary_)
    # Print the word-to-index map
    # print(vector.vocabulary_['word'])

    return text, vector.vocabulary_
#end process_data

# NOTE: I CHANGED THIS CODE SO I COULD WORK ON CONSTRUCTING THE ARCHITECTURE FOR CBOW
def generate_training_data(text, window_size=2):
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

# Build the model
class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim=100):
        super(CBOW, self).__init__()
        # Layers of the CBOW
        
        # Embedding Layer (m x d)
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        # Softmax Layer
        self.log_softmax = nn.LogSoftmax() # chose log softmax to avoid problems with multiplying probabilites and getting values too small

    def forward(self, x):
        embeddings = self.embedding(x)
        average_embeddings = torch.mean(embeddings) # gonna test it out later
        hidden_output = self.hidden(average_embeddings)
        output = self.output(hidden_output)
        return self.log_softmax(output)

    def fit(self):
        pass


def visualize_word_vectors(word_vectors):
    pass


def analyze_visualization(data):
    pass


# After Training
def visualize_word_vectors(word_vectors):
    pass


def dimensionality_reduction_with_pca(word_vectors):
    pass


def generate_scatter_plot(data):
    pass

# ====================== PREPARING THE MODEL ======================

# Reading file and creating pandas dataframe
df = pd.read_csv('shakespeare.txt', sep='\t', header=None, names=['Line'])
processed_text, vocab_list = process_data(df['Line'])

# Generating the training data
training_data = generate_training_data(text=processed_text)

# Splitting the training data into X and y pairs
X_train = [data[0] for data in training_data]
y_train = [data[1] for data in training_data]

# NOTE: TESTING TO ENSURE THAT DATA ACTUALLY WORKS - will be deleted later
print('Processed text first line: ', processed_text[0].split())
print('first six examples of training data: ', training_data[:6]) 

print('X_train first three examples: ', X_train[:3])
print('y_train first three examples: ', y_train[:3])

# Creating the CBOW model using the CBOW class
# cbow = CBOW(vocab_size=len(vocab_list))

print(len(vocab_list))
print(vocab_list['from'])