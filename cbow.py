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
    #print(vector.vocabulary_)
    # Print the word-to-index map
    #print(vector.vocabulary_['word'])

    return text, vector.vocabulary_
#end process_data

processe_text, vocab_list = process_data(df['Line'])


# Generate Training Data (may not need all functions)
def generate_training_data(text, vocab, window_size=2):
    training_data = []
    center_word_context_pair = []

    
    for sentence in text:
        
        words = sentence.split()  # split words in processed text
        indices = [vocab.get(word, -1) for word in words if word in vocab]  # turn word into index
        indices = [idx for idx in indices if idx != -1]  # get rid of words that're not in the vocabulary

        # Generate index pairs for the central and context words
        for center_word_pos in range(len(words)):
            current_context_words = []  # list to store the context words of current central word
            current_context_indices = []  # to store indices of valid context words
            
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w  # calculate the position of context word

                # check if the position of context word is available
                # skip this turn if not
                if context_word_pos < 0 or context_word_pos >= len(words) or center_word_pos == context_word_pos:
                    continue

                # Check if the context word is in vocab
                if words[context_word_pos] in vocab:
                    # Append available context words into list
                    current_context_words.append(words[context_word_pos])
                    current_context_indices.append(vocab[words[context_word_pos]])
                #end if words[context_word_pos]
            #end for w

            # Append word pairs into center_word_context only if center word is in vocab
            if words[center_word_pos] in vocab:
                center_idx = vocab[words[center_word_pos]]  # Get the index of the center word
                for context_idx in current_context_indices:
                    training_data.append((center_idx, context_idx))
                #end for context_idx
                center_word_context_pair.append((current_context_words, words[center_word_pos]))
            #end if words[center_word_pos]

        #end for center_word_pos
    #end for sentence
                
    #output a list of data pairs and an array of word pairs
    #[(index of central word, index of context word),( , ), ...]
    #[([context_word_1, context_word_2], central_word), ...]
                
    return training_data, center_word_context_pair

#end generate_training_data

#[(index of central word, index of context word),( , ), ...]
generated_index_pair, generated_center_word_context = generate_training_data(processe_text, vocab_list)

# test
for i in range(len(generated_center_word_context)):
    print(generated_center_word_context [i])
    if i > 200:
        break


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
