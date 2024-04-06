# imports
# import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Data Preparation

# Reading file and creating pandas dataframe
df = pd.read_csv('shakespeare.txt', sep='\t', header=None, names=['Line'])


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
    print(vector.vocabulary_)
    # Print the word-to-index map
    print(vector.vocabulary_['word'])


process_data(df['Line'])


# Generate Training Data (may not need all functions)
def generate_training_data(processed_documents):
    pass


# Training the model
class CBOW():
    def __init__(self, vocab_size, embedding_dim):
        pass

    def forward(self):
        pass

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
