# %% [markdown]
# # Assignment 3: Sequence-to-Sequence Modeling (TA: Kinshuk Vasisht)
# 
# In this assignment, you will perform the task of translating Indian Names to Hindi, a sequence-to-sequence modeling task, using character-level conditional language models.
# 
# As before, please make a copy of this notebook (locally or on Colab). Ensure you adhere to the guidelines and submission instructions (mentioned below) for attempting and submitting the assignment.

# %% [markdown]
# ### Guidelines for Attempting the Assignment

# %% [markdown]
# 1. Write your logic in the cells which have the comment `# ADD YOUR CODE HERE`, between the `# BEGIN CODE` and `# END CODE` comments. These cells are also demarcated by the special start (`## ==== BEGIN EVALUATION PORTION`) and end (`## ==== END EVALUATION PORTION`) comments. Do **NOT** remove any of these comments from the designated cells, otherwise your assigment may not be evaluated correctly.
# 
# 2. All imports that should be necessary are already provided as part of the notebook. Should you require additional imports, add them in the cells to be graded, but outside the `# BEGIN CODE` and `# END CODE` block. For example, if you need to import a package called `mypackage`, add it as follows in a graded cell:
# 
# ``` python
# ## ==== BEGIN EVALUATION PORTION
# 
# import mypackage # <===
# 
# def function_to_be_implemented(*args, **kwargs):
# 
#     ...
# 
#     # ADD YOUR CODE HERE
#     # BEGIN CODE
# 
#     # END CODE
# 
#     ...
# 
# ## ==== END EVALUATION PORTION
# 
# ```
# 
# 3. Only write your code in the cells designated for auto-evaluation. If you encounter any errors in the supporting cells during execution, contact the respective TAs.
# 
# 4. **Important**: Use of AI-assistive technologies such as ChatGPT or GitHub CoPilot is not permitted for this assignment. Ensure that all attempts are solely your own. Not following this rule can incur heavy penalty, including getting NO GRADE for this assignment, which will affect your grade significantly.

# %% [markdown]
# ### Submission Instructions

# %% [markdown]
# 1. Ensure your code follows all guidelines mentioned above before submission.
# 
# 2. Try to avoid any unnecessary print statements across the code. We will evaluate specific output lines which begin with the phrase `EVALUATION`. Ensure you do not modify these print statements howsoever, as they are used for auto-evaluation.
# 
# 3. When you have completely attempted the assignment, export the current notebook as a `.py` file, with the following name: `SAPName_SRNo_assignment3.py`, where `SAPName` would be your name as per SAP record, and `SRNo` will be the last 5 digits of your IISc SR number. For example, IISc student with SAP name Twyla Linda (SR no - 04-03-00-10-22-20-1-15329) would use `Twyla_Linda_15329_assignment3.py`.
# 
# 4. Once you have executed the code, certain additional files will be created. Once you are done executing all associated cells, ensure the folder structure looks as follows:
# 
# ``` python
# └─── SAPName_SRNo
#      ├─── SAPName_SRNo_assignment3.py
#      ├─── src-tokenizer
#      │    └─── tokenizer.pkl
#      ├─── tgt-tokenizer
#      │    └─── tokenizer.pkl
#      ├─── rnn.enc-dec
#      │    ├─── model.pt
#      │    ├─── loss.json
#      │    ├─── outputs.csv
#      │    └─── metadata.json
#      └─── rnn.enc-dec.attn
#           ├─── model.pt
#           ├─── loss.json
#           ├─── outputs.csv
#           └─── metadata.json
# ```
# 
# 5. Once you have validated the folder structure as above, add the exported `.py` file to the folder and submit the folder as a ZIP archive.

# %% [markdown]
# In the cell below, replace `SAPName` with your name as per SAP record, and `SRNo` with the last 5 digits of your IISc SR number. For example, IISc student with SAP name Twyla Linda (SR no - 04-03-00-10-22-20-1-15329) would use:
# 
# ```python
# STUDENT_SAP_NAME  = "Twyla Linda"
# STUDENT_SR_NUMBER = "15329"
# ```

# %%
STUDENT_SAP_NAME  = "BadrinathSinghal"
STUDENT_SR_NUMBER = "23302"

# %% [markdown]
# **Important Notes**:
# 
# - Some of the tasks in this assignment are compute intensive, and are better performed on an accelerator device (GPU, etc.). Unless you have one locally, prefer using a GPU instance on Colab for execution.
# - Due to resource restrictions on Colab, training some models may not finish in time. In such a case, ensure you store checkpoints to a persistent directory so that you may resume training once your resource limits are restored.

# %% [markdown]
# ## Outline

# %% [markdown]
# Through the last assignment, you have seen that neural language models are able to successfully capture patterns across Indian names. In this assignment, you will extend upon that idea to learn conditional language models for the task of transliteration: converting Indian names in the English alphabet to Hindi.

# %% [markdown]
# ### Marks Distribution
# 
# - Tokenization: 20 marks
# - Agnostic Task-Specific Training: 5 marks
# - Seq-2-Seq via RNN: 40 marks
# - Seq-2-Seq via RNN with Attention: 35 marks
# - Evaluation
# - (**Bonus**) Decoding Strategies: 20 marks

# %% [markdown]
# ## Setup

# %% [markdown]
# The following cells perform the basic setup such as importing the necessary packages.

# %%
# Installs packages, if using locally. Feel free to add other missing packages as required.

# %pip install tqdm nltk matplotlib numpy pandas

# %%
# Built-in imports, no installations required.

import os
import gc
import json
import math
import pickle
import subprocess
import collections
import unicodedata

# %%
# 3rd-party package imports, may require installation if not on a platform such as Colab.

import numpy
import torch
import pandas as pd
import tqdm.auto as tqdm

import matplotlib
from matplotlib import pyplot
from nltk.translate import bleu_score
from torch.utils.data import TensorDataset, DataLoader

# %%
# Please do not change anything in the following cell

# Find and load fonts that can display Hindi characters, for Matplotlib
result = subprocess.run([ 'fc-list', ':lang=hi', 'family' ], capture_output=True)
found_hindi_fonts = result.stdout.decode('utf-8').strip().split('\n')

matplotlib.rcParams['font.sans-serif'] = [
    'Source Han Sans TW', 'sans-serif', 'Arial Unicode MS',
    *found_hindi_fonts
]

# %%
# Please do not change anything in the following cell

DIRECTORY_NAME = f"{STUDENT_SAP_NAME.replace(' ', '_')}_{STUDENT_SR_NUMBER}"

os.makedirs(DIRECTORY_NAME, exist_ok=True)

# %%
def sync_vram():
    """ Synchronizes the VRAM across the GPUs, reclaiming unused memory. """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# We'll load the data for the task, which comprises of a parallel corpus of Indian Names and their Hindi equivalents.

# %%
# Make sure your code is not dependent on any of the file names as below.

# Download the training and validation datasets
# !wget -O data.train.csv "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
# !wget -O data.valid.csv "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

# %%
# Please do not change anything in the following cell

def read_dataframe(ds_type):
    """ Loads a dataframe based on the given partition type.

    Args:
        ds_type (str): Dataset type: train (train) or validation (valid)

    Returns:
        pd.DataFrame: Pandas Dataframe for the specified partition.
    """

    df = pd.read_csv(f"data.{ds_type}.csv", header=0)
    df = df[~df.isna()]
    df['Name'] = df['Name'].astype(str)
    df['Translation'] = df['Translation'].astype(str)
    return df

# Load the training and validation datasets
train_data      = read_dataframe("train")
validation_data = read_dataframe("valid")

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")

# %% [markdown]
# Here are some examples from the training dataset. Note that the dataset may be noisy so some examples may not be perfect:

# %%
train_data.sample(n=5)
# train_data['Name'].values

# %% [markdown]
# ## Tokenization

# %% [markdown]
# Now with the data, you must first prepare a tokenization strategy for feeding name pairs as a sequence to different models. For English this could be as simple as using individual characters as tokens, but Hindi has accents (मात्राएँ), a larger set of vowels (स्वर), consonants (व्यंजन), and additional composition rules (half-letters, etc.), so such a simple strategy may not be effective.
# 
# In NLP literature, multiple strategies exist for automatically learning a suitable sub-word tokenization strategy from the given data. Such tokenizers exist in two types:
# - Given a set of initial tokens, learn suitable combinations which are added as new tokens until a certain vocabulary size is reached. Examples of these include [BPE Tokenization](https://arxiv.org/abs/1508.07909) and [WordPiece Tokenization, introduced by the BERT paper](https://arxiv.org/abs/1810.04805).
# - Given a large set of initial tokens, learn suitable rules to reduce the size of the vocabulary to a desired size. An example of this includes [SentencePiece Tokenization](https://arxiv.org/abs/1808.06226).
# 
# Given empirical results, these are popular strategies to learn tokenization automatically from given data.

# %% [markdown]
# In this section, you will implement a tokenizer for the given data. There could be multiple strategies to implement tokenizers:
# - You can have a tokenizer that operates jointly over both languages or have separate tokenizers for English and Hindi.
# - Your tokenizer can learn the tokenization from data (using any one of the techniques mentioned above) or can use a fixed set of rules for decomposition.
# 
# Implement the logic for any tokenizer of your choice in the class `Tokenizer` below. Make sure to adhere to the rules and constraints in the docstrings.
# 
# The tokenizer will learn a mapping of tokens to ids and vice versa and use these to map strings. This mapping can be built based on merge rules (BPE, WordPiece, etc.) or hand-crafted rules, in the `Tokenizer.train()` function. Additionally the tokenizer will also handle preprocessing and postprocessing of strings during the encoding phase (string to tokens).

# %%
## ==== BEGIN EVALUATION PORTION
import os
import pickle
from collections import Counter


class Tokenizer:
    """ Represents the tokenizer for text data.
        Provides methods to encode and decode strings (as instance or as a batch). """

    def __init__(self):
        """ Initializes a new tokenizer.

            Any variables required in intermediate operations are declared here.
            You will also need to define things like special tokens and other things here.

            All variables declared in this function will be serialized
                and deserialized when loading and saving the Tokenizer.
            """

        # BEGIN CODE : tokenizer.init

        # ADD YOUR CODE HERE

        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.reverse_vocab = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}

        # END CODE

    @classmethod
    def load(cls, path):
        """ Loads a pre-trained tokenizer from the given directory.
           This directory will have a tokenizer.pkl file that contains all the tokenizer variables.

        Args:
            path (str): Path to load the tokenizer from.
        """
        tokenizer_file = os.path.join(path, "tokenizer.pkl")

        if not os.path.exists(path) or not os.path.exists(os.path.join(path, "tokenizer.pkl")):
            raise ValueError(cls.load.__name__ + ": No tokenizer found at the specified directory")

        with open(tokenizer_file, "rb") as ifile:
            return pickle.load(ifile)

    def save(self, path):
        """ Saves a trained tokenizer to a given directory, inside a tokenizer.pkl file.

        Args:
            path (str): Directory to save the tokenizer in.
        """

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.pkl"), 'wb') as ofile:
            pickle.dump(self, ofile)

    def train(self, data, vocab_size):
        """ Trains a tokenizer to learn meaningful representations from input data.
            In the end, learns a vocabulary of a fixed size over the given data.
            Special tokens, if any, must not be counted towards this vocabulary.

        Args:
            data (list[str]): List of input strings from a text corpus.
            vocab_size (int): Final desired size of the vocab to be learnt.
        """

        # BEGIN CODE : tokenizer.train

        # ADD YOUR CODE HERE
        self.merges = []
        word_counts = Counter()
        
        for sentence in data:
            # words = sentence.split()
            # print(words, sentence)
            for char in sentence:
                word_counts.update(char)

        word_sort = [word for word, _ in word_counts.most_common()]

        # print(word_sort[:5])

        for i, word in enumerate(word_sort[:vocab_size]):
            self.vocab[word] = i + 4
            self.reverse_vocab[i + 4] = word
        
        
        # def tokenize(self, word):
        # tokens = []
        # for char in word:
        #     if char in self.vocab:
        #         tokens.append(self.vocab[char])
        #     else:
        #         tokens.append(self.vocab['[UNK]'])
        # return tokens

        # print(vocab)
        # print(self.vocab)
        
        # print("START", len(self.vocab), len(vocab))
        
        while len(self.vocab)<vocab_size+4:
            pair_count = Counter()
            # print(data[0])
            for word in data:
                word_token = []
                for char in word:
                    if char in self.vocab:
                        word_token.append(self.vocab[char])
                    # else:
                    #     word_token.append(self.vocab['[UNK]'])
                
                for i in range(len(word_token)-1):
                    char_pair = (word_token[i], word_token[i+1])
                    pair_count[char_pair] += 1
            
            most_occ_pair = max(pair_count, key=pair_count.get)

            str_most_occ_pair = (self.reverse_vocab[most_occ_pair[0]], self.reverse_vocab[most_occ_pair[1]])
            # print(most_occ_pair, str_most_occ_pair)
            merge_token = str_most_occ_pair[0] + str_most_occ_pair[1]
            self.merges.append(most_occ_pair)
            self.vocab[merge_token] = len(self.vocab)
            self.reverse_vocab[len(self.vocab)-1] = merge_token
            # rever
            
            # print(vocab)

            new_data = []
            for word in data:
                new_word, i = [], 0
                while i < len(word):
                    if i < (len(word)-1) and word[i] == str_most_occ_pair[0] and word[i+1] == str_most_occ_pair[1]:
                        new_word.append(merge_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_data.append(new_word)
            data = new_data
            # word_sort.remove(most_occ_pair[0])
            # word_sort.remove(most_occ_pair[1])
            # word_sort.append(merge_token)

            # new_vocab = set()
            # for token in vocab:
            #     new_token = token.replace(self.reverse_vocab[most_occ_pair[0]], merge_token)
            #     new_token = new_token.replace(self.reverse_vocab[most_occ_pair[1]], merge_token)
            #     new_vocab.add(new_token)
            # vocab = new_vocab

            # self.vocab.update({token: i for i, token in enumerate(vocab)})
            # self.reverse_vocab = {i: token for token, i in self.vocab.items()}

            # print(most_occ_pair, merge_token)
            # print(self.vocab)

            # print(len(self.vocab), len(vocab))

            # print(self.vocab)

            # self.vocab[merge_token] = len(self.vocab)
            # self.reverse_vocab[len(self.reverse_vocab)] = merge_token
 
            # print(merge_token)

            # del self.reverse_vocab[self.vocab[most_occ_pair[0]]]
            # del self.reverse_vocab[self.vocab[most_occ_pair[1]]]
            # try:
            #     del self.vocab[self.reverse_vocab[most_occ_pair[0]]]
            #     del self.vocab[self.reverse_vocab[most_occ_pair[1]]]
                
            #     del self.reverse_vocab[most_occ_pair[0]]
            #     del self.reverse_vocab[most_occ_pair[1]]
            # except:
            #     continue

            


            
            # self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
            # self.reverse_vocab = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}

            # for i, word in enumerate(word_sort[:vocab_size-4]):
            #     self.vocab[word] = i + 4
            #     self.reverse_vocab[i+4] = word
        

        # END CODE


    def pad(self, tokens, length):
        """ Pads a tokenized string to a specified length, for batch processing.

        Args:
            tokens (list[int]): Encoded token string to be padded.
            length (int): Length of tokens to pad to.

        Returns:
            list[int]: Token string padded to desired length.
        """

        # BEGIN CODE : tokenizer.pad

        # ADD YOUR CODE HERE
        
        if len(tokens)>= length:
            return tokens[:length]
        else:
            token_pad = tokens + [self.vocab['[PAD]']]*(length - len(tokens))
            return token_pad

        # END CODE

    def unpad(self, tokens):
        """ Removes padding from a token string.

        Args:
            tokens (list[int]): Encoded token string with padding.

        Returns:
            list[int]: Token string with padding removed.
        """

        # BEGIN CODE : tokenizer.unpad

        # ADD YOUR CODE HERE
        unpad_tkk = []
        for tk in tokens:
            if tk != self.vocab['[PAD]']:
                unpad_tkk.append(tk)
        
        return unpad_tkk

        # END CODE

    def get_special_tokens(self):
        """ Returns the associated special tokens.

            Returns:
                dict[str, int]: Mapping describing the special tokens, if any.
                    This is a mapping between a string segment (token) and its associated id (token_id).
        """

        # BEGIN CODE : tokenizer.get_special_tokens

        # ADD YOUR CODE HERE
        special_tokens = {'[PAD]':0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}

        return special_tokens

        # END CODE

    def get_vocabulary(self):
        """ Returns the learnt vocabulary post the training process.

            Returns:
                dict[str, int]: Mapping describing the vocabulary and special tokens, if any.
                    This is a mapping between a string segment (token) and its associated id (token_id).
        """

        # BEGIN CODE : tokenizer.get_vocabulary

        # ADD YOUR CODE HERE
        return self.vocab
        # END CODE

    def encode(self, string, add_start=True, add_end=True):
        """ Encodes a string into a list of tokens.

        Args:
            string (str): Input string to be tokenized.
            add_start (bool): If true, adds the start of sequence token.
            add_end (bool): If true, adds the end of sequence token.
        Returns:
            list[int]: List of tokens (unpadded).
        """

        # BEGIN CODE : tokenizer.encode

        # ADD YOUR CODE HERE
        return_token = []

        if add_start:
            return_token.append(self.vocab['[CLS]'])
        
        for char in string:
            if char in self.vocab:
                return_token.append(self.vocab[char])
            else:
                return_token.append(self.vocab['[UNK]'])
        
        if add_end:
            return_token.append(self.vocab['[SEP]'])
        
        return return_token

        # END CODE

    def decode(self, tokens, strip_special=True):
        """ Decodes a string from a list of tokens.
            Undoes the tokenization, returning back the input string.

        Args:
            tokens (list[int]): List of encoded tokens to be decoded. No padding is assumed.
            strip_special (bool): Whether to remove special tokens or not.

        Returns:
            str: Decoded string.
        """

        # BEGIN CODE : tokenizer.decode

        # ADD YOUR CODE HERE
        
        decoded_tokens = [self.reverse_vocab[token] for token in tokens]
        
        string_decode = []
        if tokens[0] == self.vocab['[CLS]']:
            tokens = tokens[1:]
        if tokens[-1] == self.vocab['[SEP]']:
            tokens = tokens[:-1]
        
        for i, tk in enumerate(tokens):
            
            string_decode.append(self.reverse_vocab[tk])
        
        return ''.join(string_decode)
        

        # END CODE


    def batch_encode(self, batch, padding=None, add_start=True, add_end=True):
        """Encodes multiple strings in a batch to list of tokens padded to a given size.

        Args:
            batch (list[str]): List of strings to be tokenized.
            padding (int, optional): Optional, desired tokenized length. Outputs will be padded to fit this length.
            add_start (bool): If true, adds the start of sequence token.
            add_end (bool): If true, adds the end of sequence token.

        Returns:
            list[list[int]]: List of tokenized outputs, padded to the same length.
        """

        batch_output = [ self.encode(string, add_start, add_end) for string in batch ]
        if padding:
            for i, tokens in enumerate(batch_output):
                if len(tokens) < padding:
                    batch_output[i] = self.pad(tokens, padding)
        return batch_output

    def batch_decode(self, batch, strip_special=True):
        """ Decodes a batch of encoded tokens to normal strings.

        Args:
            batch (list[list[int]]): List of encoded token strings, optionally padded.
            strip_special (bool): Whether to remove special tokens or not.

        Returns:
            list[str]: Decoded strings after padding is removed.
        """
        return [ self.decode(self.unpad(tokens), strip_special=strip_special) for tokens in batch ]

## ==== END EVALUATION PORTION

# %% [markdown]
# Now with the tokenizer class, initialize and train the tokenizers for processing the parallel corpus:

# %%
## ==== BEGIN EVALUATION PORTION

# BEGIN CODE : tokenizer.create

# Initialize the tokenizers as per the desired strategy.
# ADD YOUR CODE HERE
# src_tokenizer = None
# tgt_tokenizer = None

src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()

# END CODE

## ==== END EVALUATION PORTION

# %%
## ==== BEGIN EVALUATION PORTION

# Edit the hyperparameters below as desired.
SRC_VOCAB_SIZE = 50
TGT_VOCAB_SIZE = 100

# BEGIN CODE : tokenizer.training

# Train your tokenizer(s)
# ADD YOUR CODE HERE
# print(len(src_tokenizer.vocab), src_tokenizer.vocab)
src_tokenizer.train(train_data['Name'], SRC_VOCAB_SIZE)
print("Training completed for SRC")
tgt_tokenizer.train(train_data['Translation'], TGT_VOCAB_SIZE)
print("Training completed for TGT")
# print(len(src_tokenizer.vocab), src_tokenizer.vocab)
# END CODE

## ==== END EVALUATION PORTION

# %%
# Please do not change anything in the following cell

# Save the trained tokenizers
src_tokenizer.save(os.path.join(DIRECTORY_NAME, "src_tokenizer"))
tgt_tokenizer.save(os.path.join(DIRECTORY_NAME, "tgt_tokenizer"))
# print(tgt_tokenizer.get_vocabulary())
# exit()
# %%
# Please do not change anything in the following cell

def render_glyph(token):
    """ Renders a token, handling invalid bytes in a safe, error-proof manner. """

    token = token.decode('utf-8', errors='replace') if isinstance(token, bytes) else token
    return "".join([ c if unicodedata.category(c)[0] != "C" else f"\\u{ord(c):04x}" for c in token ])

def inverse_vocabulary(tokenizer):
    """ Generates an inverse vocabulary with rendered tokens.

    Args:
        tokenizer (Tokenizer): Tokenizer whose vocabulary must be used.
    """

    return { id: render_glyph(token) for token, id in tokenizer.get_vocabulary().items() }

def apply_inverse_vocab(tokens, inv_vocab):
    """ Decodes using the given inverse vocabulary.

    Args:
        tokens (list[int]): Tokens to process.
        inv_vocab (dict[int, str]): Inverse vocabulary for mapping ids to tokens.

    Returns:
        list[str]: Mapped token glyphs.
    """

    return [ inv_vocab[id] for id in tokens ]

# %% [markdown]
# We visualize a few outputs of the learnt tokenizers to assess their working:

# %%
# Please do not change anything in the following cell

src_id_to_token = inverse_vocabulary(src_tokenizer)

for example in train_data['Name'].sample(n=5, random_state=20240227):
    print("Name           :", example)
    tokens = src_tokenizer.encode(example)
    print("Tokens         :", tokens)
    print("Tokens (glyphs):", apply_inverse_vocab(tokens, src_id_to_token))
    print("Decoded        :", src_tokenizer.decode(tokens), end='\n\n')

# %%
# Please do not change anything in the following cell

tgt_id_to_token = inverse_vocabulary(tgt_tokenizer)

for example in train_data['Translation'].sample(n=5, random_state=20240227):
    print("Name           :", example)
    tokens = tgt_tokenizer.encode(example)
    print("Tokens         :", tokens)
    print("Tokens (glyphs):", apply_inverse_vocab(tokens, tgt_id_to_token))
    print("Decoded        :", tgt_tokenizer.decode(tokens), end='\n\n')

# %%
# Please do not change anything in the following cell

# Run some tests to ensure your tokenizer implementation works as intended.

instances = train_data['Translation'].sample(n=5, random_state=20240227).tolist()

try:
    max_length = max(map(
        lambda x: len(tgt_tokenizer.encode(x, add_start=False, add_end=False)),
        instances
    ))

    # Batch encode all instances with 'max' padding':
    tokenized_instances = tgt_tokenizer.batch_encode(
        instances, padding=max_length, add_start=False, add_end=False
    )

    # Check if length of encoded strings is consistent with the expected length.
    assert all(len(tok_str) == max_length for tok_str in tokenized_instances)

except AssertionError:
    print((
        "[!] Your tokenizer does not encode strings correctly, "
        "ensure you have implemented padding appropriately!"
    ))

try:
    max_length = max(map(
        lambda x: len(tgt_tokenizer.encode(x, add_start=True, add_end=True)),
        instances
    ))

    # Batch encode all instances with 'max' padding':
    tokenized_instances = tgt_tokenizer.batch_encode(
        instances, padding=max_length, add_start=True, add_end=True
    )

    # Check if length of encoded strings is consistent with the expected length.
    assert all(len(tok_str) == max_length for tok_str in tokenized_instances)

    # # Check if all strings start with the correct 'start' tokens.
    assert all(tok_str[0] == tokenized_instances[0][0] for tok_str in tokenized_instances)

    # # Check if all strings end with the correct 'end' tokens.
    end_i = [ i for i, seq in enumerate(tokenized_instances) if len(tgt_tokenizer.unpad(seq)) == max_length ]
    pad_i = [ i for i, seq in enumerate(tokenized_instances) if len(tgt_tokenizer.unpad(seq)) <  max_length ]

    assert all(
        tokenized_instances[i][-1] == tokenized_instances[end_i[0]][-1]
        for i in end_i
    )
    assert all(
        tokenized_instances[i][-1] == tokenized_instances[pad_i[0]][-1]
        for i in pad_i
    )
    pad_lengths = [ tokenized_instances[i].index(tokenized_instances[end_i[0]][-1]) for i in pad_i ]
    assert all(
        all(tok == tokenized_instances[pad_i[0]][-1] for tok in tokenized_instances[i][plen+1:])
        for i, plen in zip(pad_i, pad_lengths)
    )

except AssertionError:
    print((
        "[!] Your tokenizer does not encode strings correctly, "
        "ensure you have used start and end tokens appropriately!"
    ))

# %% [markdown]
# We now abstract away the tokenizer into a pytorch compatible TokenizedDataset that will handle the tokenization internally:

# %%
# Please do not change anything in the following cell

class TokenizerDataset(TensorDataset):
    """ Abstraction of the tokenizer functions as a pytorch dataset. """

    def __init__(self, data, src_tokenizer, tgt_tokenizer, src_padding=None, tgt_padding=None):
        """ Initializes the dataset.

        Args:
            data: DataFrame of input and output strings.
            src_tokenizer (Tokenizer): Tokenizer for the source language.
            tgt_tokenizer (Tokenizer): Tokenizer for the target language.
            src_padding (int, optional): Padding length for the source text. Defaults to None.
            tgt_padding (int, optional): Padding length for the target text. Defaults to None.
        """

        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_padding = src_padding
        self.tgt_padding = tgt_padding

    def collate(self, batch):
        """ Collates data instances into a batch of tokenized tensors.

        Args:
            batch (list[tuple]): List of x, y pairs.

        Returns:
            tuple[torch.Tensor|PackedSequence, torch.Tensor|PackedSequence]: pair of tokenized tensors.
        """

        x_batch = [ data[0] for data in batch ]
        y_batch = [ data[1] for data in batch ]

        x_batch = self.src_tokenizer.batch_encode(x_batch, self.src_padding)
        y_batch = self.tgt_tokenizer.batch_encode(y_batch, self.tgt_padding)

        if self.src_padding is None:
            x_batch = torch.nn.utils.rnn.pack_sequence([ torch.tensor(tokens) for tokens in x_batch ], False)
        else:
            x_batch = torch.tensor(x_batch)

        if self.tgt_padding is None:
            y_batch = torch.nn.utils.rnn.pack_sequence([ torch.tensor(tokens) for tokens in y_batch ], False)
        else:
            y_batch = torch.tensor(y_batch)

        return x_batch, y_batch

    def __getitem__(self, index):
        """ Returns the nth instance from the dataset.

        Args:
            index (int): Index of the instance to retrieve.

        Returns:
            tuple[str, str]: Untokenized instance pair.
        """

        return (
            self.data['Name'][index],
            self.data['Translation'][index]
        )

    def __len__(self):
        """ Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

# %% [markdown]
# ## Model-Agnostic Training

# %% [markdown]
# Next, you'll implement a Trainer to train different models, since the data and tokenizer remains the same for all models.
# 
# This trainer will receive the model, a loss function, an optimizer, a training and (optionally) a validation dataset and use these to train (and validate) the model.
# 
# The trainer will also take care of handling checkpoints for training, which can be used to resume training across sessions.
# 
# Derived classes can also be defined to handle different architectures, as to be done in the model-specific classes below.

# %%
## ==== BEGIN EVALUATION PORTION

class Trainer:
    """ Performs model training in a model-agnostic manner.
        Requires specifying the model instance, the loss criterion to optimize,
          the optimizer to use and the directory to save data to.
    """

    def __init__(self, directory, model, criterion, optimizer):
        """ Initializes the trainer.

        Args:
            directory (str): Directory to save checkpoints and the model data in.
            model (torch.nn.Module): Torch model (must inherit `torch.nn.Module`) to train.
            criterion (torch.nn.Function): Loss criterion, i.e., the loss function to optimize for training.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
        """

        self.model            = model
        self.optimizer        = optimizer
        self.criterion        = criterion
        self.directory        = directory
        self.last_checkpoint  = 0
        self.loss_history     = { 'train': [], 'valid': [] }

        os.makedirs(self.directory, exist_ok=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = self.criterion.to(self.device)


    @staticmethod
    def make_dataloader(dataset, shuffle_data=True, batch_size=8, collate_fn=None):
        """ Create a dataloader for a torch Dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to process.
            shuffle_data (bool, optional): If true, shuffles the data. Defaults to True.
            batch_size (int, optional): Number of items per batch. Defaults to 8.
            collate_fn (function, optional): Function to use for collating instances to a batch.

        Returns:
            torch.utils.data.DataLoader: Dataloader over the given data, post processing.
        """

        # BEGIN CODE : trainer.make_dataloader

        # ADD YOUR CODE HERE

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, collate_fn=collate_fn)

        # END CODE

    def train_step(self, x_batch, y_batch):
        """ Performs a step of training, on the training batch.

        Args:
            x_batch (torch.Tensor): Input batch.
            y_batch (torch.Tensor): Output batch.

        Returns:
            float: Training loss with the current model, on this batch.
        """

        # BEGIN CODE : trainer.train_step

        # ADD YOUR CODE HERE
        self.model.train()

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        self.model = self.model.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(x_batch)


        loss = self.criterion(outputs, y_batch)
        loss.backward()
        
        self.optimizer.step()

        return loss.item()

        # END CODE

    def eval_step(self, validation_dataloader):
        """ Perfoms an evaluation step, on the validation dataloader.

        Args:
            validation_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.

        Returns:
            float: Validation loss with the current model checkpoint.
        """

        # BEGIN CODE : trainer.eval_step

        # ADD YOUR CODE HERE

        self.model.eval()

        total_loss = 0.0
        num_batch = 0

        with torch.no_grad():
            for x_batch, y_batch in validation_dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)

                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                num_batch += 1
        
        avg_loss = total_loss/num_batch

        return avg_loss

        # END CODE

    def train(self, train_dataset, validation_dataset=None,
              num_epochs=10, batch_size=8, shuffle=True,
              save_steps=100, eval_steps=100, collate_fn=None):
        """ Handles the training loop for the model.

        Args:
            train_dataset (torch.utils.data.Dataset): Dataset to train on.
            validation_dataset (torch.utils.data.Dataset, optional): Data to validate on. Defaults to None.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 10.
            batch_size (int, optional): Number of items to process per batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data or not. Defaults to True.
            save_steps (int, optional): Number of steps post which a checkpoint should be saved. Defaults to 100.
            eval_steps (int, optional): Number of steps post which the model should be evaluated. Defaults to 100.
            collate_fn (function, optional): Function to use for collating instances to a batch.
        """

        current_checkpoint = 0
        self.model.to(self.device)
        self.model.train()

        with tqdm.tqdm(total = math.ceil(len(train_dataset) / batch_size) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                train_dataloader      = self.make_dataloader(train_dataset, shuffle, batch_size, collate_fn)
                if validation_dataset is not None:
                    validation_dataloader = self.make_dataloader(validation_dataset, shuffle, batch_size, collate_fn)

                for batch, (x_batch, y_batch) in enumerate(train_dataloader):
                    pbar.set_description(f"Epoch {epoch+1} / {num_epochs}, Batch {batch}")

                    # If we are resuming training, skip this iteration
                    if current_checkpoint < self.last_checkpoint:
                        current_checkpoint += 1
                        pbar.update()
                        continue

                    # Do a step of training
                    loss = self.train_step(x_batch, y_batch)
                    self.loss_history['train'].append(loss)
                    pbar.set_postfix({ 'batch': batch+1, 'loss': loss })

                    current_checkpoint += 1
                    pbar.update()

                    # Evaluate after every eval_steps
                    if (current_checkpoint) % eval_steps == 0:
                        if validation_dataset is not None:
                            val_loss = self.eval_step(validation_dataloader)
                            self.loss_history['valid'].append(val_loss)
                        else:
                            val_loss = None

                        print('[>]', f"epoch #{epoch+1:{len(str(num_epochs))}},",
                              f"batch #{batch+1:{len(str(len(train_dataloader)))}}:",
                              "loss:", f"{loss:.8f}", '|', "val_loss:", f"{val_loss:.8f}")

                    # Save after every save_steps
                    if (current_checkpoint) % save_steps == 0:
                        self.save(current_checkpoint, { 'loss': loss, 'checkpoint': current_checkpoint })

                    # free unused resources
                    # sync_vram()

            self.save(current_checkpoint)

    def resume(self):
        """ Resumes training session from the most recent checkpoint. """

        if checkpoints := os.listdir(self.directory):
            self.last_checkpoint = max(map(lambda x: int(x[11:]), filter(lambda x: 'checkpoint-' in x, checkpoints)))
            checkpoint_dir = os.path.join(self.directory, f"checkpoint-{self.last_checkpoint}")
            self.model.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                map_location=self.device
            ))
            self.model.to(self.device)
            self.optimizer.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "optimizer.pt"),
                map_location=self.device
            ))
            with open(os.path.join(checkpoint_dir, "loss.json"), 'r', encoding='utf-8') as ifile:
                self.loss_history = json.load(ifile)

    def save(self, checkpoint=None, metadata=None):
        """ Saves an associated model or a training checkpoint.

            If a checkpoint is specified, saves a checkpoint specific directory with optimizer data
                so that training can be resumed post that checkpoint.

        Args:
            checkpoint (int, optional): Checkpoint index. Defaults to None.
            metadata (dict[str, any], optional): Additional metadata to save alongside a checkpoint. Defaults to None.
        """

        if checkpoint is not None:
            checkpoint_dir = os.path.join(self.directory, f"checkpoint-{checkpoint}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            with open(os.path.join(checkpoint_dir, "loss.json"), "w+", encoding='utf-8') as ofile:
                json.dump(self.loss_history, ofile, ensure_ascii=False, indent=2)
            if metadata:
                with open(os.path.join(checkpoint_dir, "metadata.json"), "w+", encoding='utf-8') as ofile:
                    json.dump(metadata, ofile, ensure_ascii=False, indent=2)
        else:
            torch.save(self.model, os.path.join(self.directory, "model.pt"))
            with open(os.path.join(self.directory, "loss.json"), "w+", encoding='utf-8') as ofile:
                json.dump(self.loss_history, ofile, ensure_ascii=False, indent=2)
            if metadata:
                with open(os.path.join(self.directory, "metadata.json"), "w+", encoding='utf-8') as ofile:
                    json.dump(metadata, ofile, ensure_ascii=False, indent=2)

## ==== END EVALUATION PORTION

# %% [markdown]
# 
# 
# ```
# # This is formatted as code
# ```
# 
# To test that the trainer works, try training a simple MLP network:

# %%
X_train = torch.rand((500, 2))                      # (N x 2)
X_dev   = torch.rand((20 , 2))                      # (N x 2)

Y_train = (X_train[:, 0] - X_train[:, 1])[:, None]  # (N x 1)
Y_dev   = (X_dev  [:, 0] - X_dev  [:, 1])[:, None]  # (N x 1)

dummy_train_dataset = TensorDataset(X_train, Y_train)
dummy_val_dataset   = TensorDataset(X_dev  , Y_dev  )

# %%
model = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

trainer = Trainer("mlp", model, loss_fn, optimizer)
trainer.train(dummy_train_dataset, dummy_val_dataset, batch_size=10, save_steps=300)

# %% [markdown]
# ## Seq-2-Seq Modeling with RNNs

# %% [markdown]
# In this section, you will implement an encoder-decoder network using RNNs, to learn a conditional language model for the task of translating the names to Hindi.
# 
# You can use any type of RNN for this purpose: `RNN`, `GRU`, `LSTM`, etc. Consult the pytorch documentation for additional information.
# 
# Additional tips for training:
# - Use regularization: Dropout, etc.
# - Use a suitable optimizer, such as Adam.
# - Format data accordingly before passing it to the trainer, using the helper functions.
# - Do you need to pad sequences when processing inputs as a batch?

# %%
## ==== BEGIN EVALUATION PORTION
import torch.nn as nn


class RNNEncoderDecoderLM(torch.nn.Module):
    """ Implements an Encoder-Decoder network, using RNN units. """

    # Feel free to add additional parameters to __init__
    def __init__(self, src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        """ Initializes the encoder-decoder network, implemented via RNNs.

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embd_dims (int): Embedding dimensions.
            hidden_size (int): Size/Dimensions for the hidden states.
        """

        super(RNNEncoderDecoderLM, self).__init__()

        # Dummy parameter to track the model device. Do not modify.
        self._dummy_param = torch.nn.Parameter(torch.Tensor(0), requires_grad=False)

        # BEGIN CODE : enc-dec-rnn.init

        # # # ADD YOUR CODE HERE
        self.hidden_size = hidden_size
        self.enc_embedding = nn.Embedding(src_vocab_size, embd_dims)
        self.enc_gru = nn.GRU(embd_dims, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)


        # Decoder model initialization 
        self.dec_embedding = nn.Embedding(tgt_vocab_size, embd_dims)
        self.dec_gru = nn.GRU(embd_dims, hidden_size, num_layers, batch_first=True)
        self.dec_out = nn.Linear(hidden_size, tgt_vocab_size)
        # END CODE

    @property
    def device(self):
        """ Returns the device the model parameters are on. """
        return self._dummy_param.device

    def forward(self, inputs, decoder_inputs, decoder_hidden_state=None):
        """ Performs a forward pass over the encoder-decoder network.

            Accepts inputs for the encoder, inputs   for the decoder, and hidden state for the decoder to continue generation after the given input.

        Args:
            inputs (torch.Tensor): tensor of shape [batch_size?, max_seq_length]
            decoder_inputs (torch.Tensor): tensor of shape [batch_size?, 1]
            decoder_hidden_state (any): tensor to represent decoder hidden state from time step T-1.

        Returns:
            tuple[torch.Tensor, any]: output from the decoder, and associated hidden state for the next step.
            Decoder outputs should be log probabilities over the target vocabulary.
        """

        # BEGIN CODE : enc-dec-rnn.forward

        # ADD YOUR CODE HERE
        decoder_embedded = self.dec_embedding(decoder_inputs)
        
        if decoder_hidden_state is not None:
            
            decoder_output, decoder_hidden = self.dec_gru(decoder_embedded, decoder_hidden_state)
        else:
            lengths = torch.count_nonzero(inputs, dim=-1).cpu()
            encoder_embedded = self.enc_embedding(inputs)
            if len(lengths.shape)>0:
                encoder_embedded = torch.nn.utils.rnn.pack_padded_sequence(encoder_embedded, lengths=lengths, batch_first=True, enforce_sorted=False)
            encoder_output, encoder_hidden = self.enc_gru(encoder_embedded)
            # print(decoder_embedded.shape, encoder_hidden.shape)
            # enc_hidd = encoder_hidden[:,-1,:].unsqueeze(1)
            decoder_output, decoder_hidden = self.dec_gru(decoder_embedded, encoder_hidden)

        # output = self.dropout(self.dec_out(decoder_output.squeeze(1)))
        output = self.dec_out(decoder_output.squeeze(1))

        # print("OP", output.shape)

        return output, decoder_hidden
        # END CODE

    def log_probability(self, seq_x, seq_y):
        
        """ Compute the conditional log probability of seq_y given seq_x, i.e., log P(seq_y | seq_x).
        
        Args:
            seq_x (torch.tensor): Input sequence of tokens.
            seq_y (torch.tensor): Output sequence of tokens.

        Returns:
            float: Log probability of seq_y given seq_x
        """

        # BEGIN CODE : enc-dec-rnn.log_probability

        # ADD YOUR CODE HERE

        encoder_hidden = None
        encoder_embedded = self.enc_embedding(seq_x)
        encoder_outputs, encoder_hidden = self.enc_gru(encoder_embedded, encoder_hidden)

        # Initialize decoder hidden state with final encoder hidden state
        decoder_hidden = encoder_hidden

        # Initialize log probabilities
        log_prob = 0.0

        # Decode sequence one step at a time
        for i in range(seq_y.size(1)):
            # Prepare decoder input
            decoder_input = seq_y[:, i:i+1]

            # Embed decoder input
            decoder_embedded = self.dec_embedding(decoder_input)

            # Decode one step
            decoder_output, decoder_hidden = self.dec_gru(decoder_embedded, decoder_hidden)

            # Compute logits
            logits = self.dec_out(decoder_output).squeeze(1)

            # Compute log probabilities using softmax
            log_probs_step = F.log_softmax(logits, dim=1)

            # Gather log probability of true target tokens
            true_tokens = seq_y[:, i]
            log_prob_step = log_probs_step.gather(1, true_tokens.unsqueeze(1))

            # Accumulate log probabilities
            log_prob += log_prob_step.squeeze()

        # Average log probabilities over sequence length
        log_prob /= seq_y.size(1)

        return log_prob.sum().item()


        # END CODE

## ==== END EVALUATION PORTION

# %% [markdown]
# To train the above model, implement for training and evaluation steps in the `RNNEncoderDecoderTrainer` class below:

# %%
## ==== BEGIN EVALUATION PORTION

class RNNEncoderDecoderTrainer(Trainer):
    """ Performs model training for RNN-based Encoder-Decoder models. """

    def __init__(self, directory, model, criterion, optimizer):
        """ Initializes the trainer.

        Args:
            directory (str): Directory to save checkpoints and the model data in.
            model (torch.nn.Module): Torch model to train.
            criterion (torch.nn.Function): Loss Criterion.
            optimizer (torch.optim.Optimizer): Optimizer to use.
        """

        super(RNNEncoderDecoderTrainer, self).__init__(directory, model, criterion, optimizer)

    @staticmethod
    def make_dataloader(dataset, shuffle_data=True, batch_size=8, collate_fn=None):
        """ Create a dataloader for a torch Dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to process.
            shuffle_data (bool, optional): If true, shuffles the data. Defaults to True.
            batch_size (int, optional): Number of items per batch. Defaults to 8.
            collate_fn (function, optional): Function to collate instances in a batch.

        Returns:
            torch.utils.data.DataLoader: Dataloader over the given data, post processing.
        """

        # BEGIN CODE : rnn-enc-dec-trainer.make_dataloader

        # ADD YOUR CODE HERE
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, collate_fn=collate_fn)
        # END CODE

    def train_step(self, x_batch, y_batch):
        """ Performs a step of training, on the training batch.

        Args:
            x_batch (torch.Tensor): Input batch tensor, of shape [batch_size, *instance_shape].
              For RNNs this is [batch_size, src_padding] or a torch.nn.utils.rnn.PackedSequence of varying lengths per batch (depends on padding).
            y_batch (torch.Tensor): Output batch tensor, of shape [batch_size, *instance_shape].
              For RNNs this is [batch_size, tgt_padding] or a torch.nn.utils.rnn.PackedSequence of varying lengths per batch (depends on padding).

        Returns:
            float: Training loss with the current model, on this batch.
        """

        # BEGIN CODE : rnn-enc-dec-trainer.train_step

        # ADD YOUR CODE HERE
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # loss = None
        # decoder_hidden_state = None

        # Forward pass of the encoder and the decoder model 
        # for i in range(y_batch.size(1)-1):
        #     decoder_outputs, decoder_hidden_state = self.model(x_batch, y_batch[:,i:i+1], decoder_hidden_state)
        #     # print(decoder_outputs.shape, y_batch[:,i+1].shape)
        #     # print(decoder_outputs.shape,  y_batch[:,i+1].shape)
        #     if loss is None:
        #         loss = self.criterion(decoder_outputs, y_batch[:,i+1])
        #     else:
        #         loss += self.criterion(decoder_outputs, y_batch[:,i+1])
        #     # print(decoder_outputs.shape)
        #     # raise ValueError()

        # # print(decoder_outputs.shape, y_batch.shape)
        # # print(y_batch)
        # # print(y_batch.view(-1).shape, decoder_outputs.view(-1, decoder_outputs.size(-1)).shape)
        # # loss = self.criterion(
        #     # decoder_outputs.view(-1, decoder_outputs.size(-1)),
        #     # y_batch.view(-1)
        # # )
        # loss /= (y_batch.size(1)-1)

        y_batch_pred, decoder_hidden_state = [], None

        for i in range(y_batch.shape[1]-1):
            logits, decoder_hidden_state = self.model(
                x_batch.to(self.device),
                y_batch[:, i:i+1].to(self.device),
                decoder_hidden_state
            )
            y_batch_pred.append(logits)

        y_batch_pred = torch.stack(y_batch_pred, dim=1)

        y_batch_pred = y_batch_pred.reshape(-1, y_batch_pred.shape[-1])
        y_batch      = y_batch[:, 1:].reshape(-1).to(self.device)

        loss = self.criterion(y_batch_pred, y_batch)
        loss [y_batch == 0] = 0

        loss = loss.sum() / (y_batch != 0).sum()

        loss.backward()

        self.optimizer.step()
        # raise ValueError()


        # total_loss += loss.item()
        return loss.item()

        # END CODE

    def eval_step(self, validation_dataloader):
        """ Perfoms an evaluation step, on the validation dataloader.

        Args:
            validation_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.

        Returns:
            float: Validation loss with the current model checkpoint.
        """

        # BEGIN CODE : rnn-enc-dec-trainer.eval_step

        # ADD YOUR CODE HERE

        self.model.eval()
        
        total_loss = 0
        num_batch = 0 

        with torch.no_grad():
            for x_batch, y_batch in validation_dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)


                decoder_outputs, _ = self.model(x_batch, y_batch)

                loss = self.criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1))[:-1,:],
                    y_batch.view(-1)[1:]
                )

                total_loss += loss.item()
                num_batch += 1

        return total_loss / num_batch   
        # END CODE

## ==== END EVALUATION PORTION

# %%
## == BEGIN EVALUATION PORTION

# Edit the hyperparameters below to your desired values.

# BEGIN CODE : rnn-enc-dec.params

# Add parameters related to the model here.
rnn_enc_dec_params = {
    'src_vocab_size': SRC_VOCAB_SIZE,
    'tgt_vocab_size': TGT_VOCAB_SIZE,
    'embd_dims'     : 256,
    'hidden_size'   : 512,
    'dropout'       : 0.1,
    'num_layers'    : 2
}

# Add parameters related to the dataset processing here.
rnn_enc_dec_data_params = dict(
    src_padding=26,
    tgt_padding=27,
)

# Add parameters related to training here.
rnn_enc_dec_training_params = dict(
    num_epochs=20,
    batch_size=16,
    shuffle=True,
    save_steps=1000,
    eval_steps=100000000000000
)

# END CODE

# Do not forget to set a deterministic seed.
torch.manual_seed(42)

model = RNNEncoderDecoderLM(**rnn_enc_dec_params)

# BEGIN CODE : rnn-enc-dec.train

# ADD YOUR CODE HERE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# END CODE

trainer = RNNEncoderDecoderTrainer(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec"),
    model, criterion, optimizer
)

## == END EVALUATION PORTION

# %%
# Please do not change anything in the following cell.

train_dataset      = TokenizerDataset(train_data     , src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)
validation_dataset = TokenizerDataset(validation_data, src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)


rnn_enc_dec_train_data = dict(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    collate_fn=train_dataset.collate
)

# Resume training from the last checkpoint, if interrupted midway, else begins training from scratch.
trainer.resume()

# Train as per specified training parameters.
trainer.train(**rnn_enc_dec_train_data, **rnn_enc_dec_training_params)

# %%

# exit()
# %%
# trainer.resume()
# print("Trainer check: ",trainer.last_checkpoint)
# exit()

# Train as per specified training parameters.
# trainer.train(**rnn_enc_dec_train_data, **rnn_enc_dec_training_params)

# %%
# Please do not change anything in the following cell.

# Save the final model, with additional metadata.
trainer.save(metadata={
    'model'   : rnn_enc_dec_params,
    'data'    : rnn_enc_dec_data_params,
    'training': rnn_enc_dec_training_params
})

# %% [markdown]
# To validate training, look at sample translations for different examples, and probabilities assigned to different outputs.
# 
# Extensive evaluation and comparison against other approaches will be carried out later.

# %%
def rnn_greedy_generate(model, seq_x, src_tokenizer, tgt_tokenizer, max_length):
    """ Given a source string, translate it to the target language using the trained model.
        This function should perform greedy sampling to generate the results.

    Args:
        model (nn.Module): RNN Type Encoder-Decoder Model
        seq_x (str): Input string to translate.
        src_tokenizer (Tokenizer): Source language tokenizer.
        tgt_tokenizer (Tokenizer): Target language tokenizer.
        max_length (int): Maximum length of the target sequence to decode.

    Returns:
        str: Generated string for the given input in the target language.
    """

    # BEGIN CODE : enc-dec-rnn.greedy_generate

    # ADD YOUR CODE HERE
    model.eval()
    seq_x_tokens = src_tokenizer.encode(seq_x, add_start=True, add_end=True)
    seq_x_tokens = torch.tensor(seq_x_tokens).unsqueeze(0).to(model.device)  # Add batch dimension

    dec_input = torch.tensor([[tgt_tokenizer.vocab['[CLS]']]]).to(model.device)  # Start token
    dec_hidden = None

    output_tokens = []  # List to store generated output tokens

    # Generate output tokens iteratively
    for _ in range(max_length):
        # Forward pass through the model
        logits, dec_hidden = model(seq_x_tokens, dec_input, dec_hidden)
        next_token_idx = torch.argmax(logits, dim=-1)
        output_tokens.append(next_token_idx.item())
        # dec_input = torch.tensor([[tgt_tokenizer.vocab['[CLS]']]]).to(model.device)
        dec_input = torch.tensor([[next_token_idx.item()]]).to(model.device)
        if next_token_idx.item() == tgt_tokenizer.vocab['[SEP]']:
            break

    generated_string = tgt_tokenizer.decode(output_tokens)
    return generated_string

    # END CODE

# %%
# Please do not change anything in the following cell.

for _, row in train_data.sample(n=5, random_state=42).iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )

    print("Name                      :", row['Name'])
    print("Translation (Expected)    :", row['Translation'])
    print("Translation (Model)       :", y_pred)

    print()

# %%
# Please do not change anything in the following cell.

for _, row in validation_data.sample(n=5, random_state=42).iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )

    print("Name                      :", row['Name'])
    print("Translation (Expected)    :", row['Translation'])
    print("Translation (Model)       :", y_pred)

    print()

# %%
# Please do not change anything in the following cell.

output_data = []
for _, row in validation_data.iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )
    output_data.append({ 'Name': row['Name'], 'Translation': y_pred })

pd.DataFrame.from_records(output_data).to_csv(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec", "outputs.csv"), index=False
)

# %%
# Please do not change anything in the following cell.

# Release resources
if 'trainer' in globals():
    del trainer

if 'model' in globals():
    del model

sync_vram()

# %% [markdown]
# ## Seq-2-Seq Modeling with RNN + Attention

# %% [markdown]
# In this module, you'll augment the Encoder-Decoder architecture to utilize attention, by implementing an Attention module that attends over the representations / inputs from the encoder.
# 
# Many approaches have been proposed in literature towards implementing attention. You are free to explore and use any implementation of your choice.
# 
# Some popular approaches are desribed in the original [paper by Bahdanau et al., 2014 on NMT](https://arxiv.org/abs/1409.0473) and an [exploratory paper by Luong et al, 2015](https://arxiv.org/abs/1508.04025) which explores different effective approaches to attention, including global and local attention.

# %%
## ==== BEGIN EVALUATION PORTION
import torch.nn.functional as F


class AttentionModule(torch.nn.Module):
    """ Implements an attention module """

    # Feel free to add additional parameters to __init__
    def __init__(self, input_size):
        """ Initializes the attention module.
            Feel free to declare any parameters as required. """

        super(AttentionModule, self).__init__()

        # BEGIN CODE : attn.init

        # ADD YOUR CODE HERE
        self.Wa = nn.Linear(input_size, input_size)
        self.Ua = nn.Linear(input_size, input_size)
        self.Va = nn.Linear(input_size, 1)

        # END CODE

    def forward(self, encoder_outputs, decoder_hidden_state, attn_mask=None):
        """ Performs a forward pass over the module, computing attention scores for inputs.

        Args:
            encoder_outputs (torch.Tensor): Output representations from the encoder, of shape [batch_size?, src_seq_len, output_dim].
            decoder_hidden_state (torch.Tensor): Hidden state from the decoder at current time step, of appropriate shape as per RNN unit (with optional batch dim).

        Returns:
            torch.Tensor: Attentions scores for given inputs, of shape [batch_size?, 1, src_seq_len]
        """

        # BEGIN CODE : attn.forward

        # ADD YOUR CODE HERE
        # print(encoder_outputs.shape, decoder_hidden_state.shape)
        decoder_hidden_state_transformed = self.Ua(decoder_hidden_state).transpose(0, 1)[:, 0:1, :]

        scores = self.Va(torch.tanh(self.Wa(encoder_outputs) + decoder_hidden_state_transformed))
        scores = scores.squeeze(2).unsqueeze(1)

        if attn_mask is not None:
            scores[~(attn_mask.unsqueeze(1))] = float("-inf")

        weights = F.softmax(scores, dim=-1)
        # context = torch.bmm(weights, keys)

        return weights

        # END CODE

## ==== END EVALUATION PORTION

# %%
## ==== BEGIN EVALUATION PORTION


class RNNEncoderDecoderLMWithAttention(torch.nn.Module):
    """ Implements an Encoder-Decoder network, using RNN units, augmented with attention. """

    # Feel free to add additional parameters to __init__
    def __init__(self,src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        """ Initializes the encoder-decoder network, implemented via RNNs.

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embd_dims (int): Embedding dimensions.
            hidden_size (int): Size/Dimensions for the hidden states.
        """

        super(RNNEncoderDecoderLMWithAttention, self).__init__()

        # Dummy parameter to track the model device. Do not modify.
        self._dummy_param = torch.nn.Parameter(torch.Tensor(0), requires_grad=False)

        # BEGIN CODE : enc-dec-rnn-attn.init

        # ADD YOUR CODE HERE

        self.hidden_size = hidden_size
        self.enc_embedding = nn.Embedding(src_vocab_size, embd_dims)
        self.enc_gru = nn.GRU(embd_dims, hidden_size, num_layers, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        self.attention = AttentionModule(hidden_size)


        # Decoder model initialization 
        self.dec_embedding = nn.Embedding(tgt_vocab_size, embd_dims)
        self.dec_gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dec_out = nn.Linear(hidden_size, tgt_vocab_size)
        self.SOS_token = 0

        # END CODE

    @property
    def device(self):
        return self._dummy_param.device

    def log_probability(self, seq_x, seq_y):
        """ Compute the conditional log probability of seq_y given seq_x, i.e., log P(seq_y | seq_x).

        Args:
            seq_x (torch.tensor): Input sequence of tokens, of shape [src_seq_len] (no batch dim)
            seq_y (torch.tensor): Output sequence of tokens, of shape [tgt_seq_len] (no batch dim)

        Returns:
            float: Log probability of generating sequence y, given sequence x.
        """

        # BEGIN CODE : enc-dec-rnn-attn.probability

        # ADD YOUR CODE HERE
        src_embedded = self.enc_embedding(seq_x)
        tgt_embedded = self.dec_embedding(seq_y)

        # Encode source sequence
        encoder_outputs, (encoder_hidden, _) = self.enc_gru(src_embedded.unsqueeze(1))

        # Initialize decoder hidden state with final encoder hidden state
        decoder_hidden, _ = encoder_hidden

        # Compute attention scores for all time steps
        attention_weights = self.attention(encoder_outputs, decoder_hidden)

        # Compute context vectors for all time steps
        context = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1)).squeeze(1)

        # Concatenate target embeddings with context vectors
        decoder_inputs = torch.cat((tgt_embedded, context.unsqueeze(0).repeat(seq_y.size(0), 1, 1)), dim=2)

        # Initialize decoder hidden state
        decoder_hidden_states = torch.zeros(seq_y.size(0), 1, self.hidden_size, device=self.device)

        # Apply decoder LSTM to all time steps
        decoder_outputs, _ = self.dec_gru(decoder_inputs, (decoder_hidden_states, decoder_hidden_states))

        # Apply dropout
        # decoder_outputs = self.dropout(decoder_outputs)

        # Compute output logits
        logits = self.dec_out(decoder_outputs)

        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log probabilities of target tokens
        log_probs_y = torch.gather(log_probs, 2, seq_y.unsqueeze(2)).squeeze(2)

        # Sum log probabilities along time steps and return total log probability
        return log_probs_y.sum()

        # END CODE

    def attentions(self, seq_x, terminate_token, max_length):
        """ Obtain attention over a sequence for decoding to the target language.

        Args:
            seq_x (torch.tensor): Tensor representing the source sequence, of shape [src_seq_len] (no batch dim)
            terminate_token (int): Token to use as EOS, to stop generating outputs.
            max_length (int): Maximum length to use to terminate the sampling.

        Returns:
            tuple[torch.tensor, torch.tensor]:
                A tuple of two tensors: the attentions over individual output tokens ([tgt_seq_len, src_seq_len])
                and the best output tokens ([tgt_seq_len]) per sequence step, based on greedy sampling.
        """

        # BEGIN CODE : rnn-enc-dec-attn.attentions

        # ADD YOUR CODE HERE

        batch_size = 1
        seq_len = len(seq_x)
        # print("Seq_x : ", seq_x.shape)
        seq_x = seq_x.to(self.device)
        if len(seq_x.shape) < 2:
            seq_x = seq_x.unsqueeze(0)

        # Initialize tensors to store attention weights and output tokens
        attentions = []  # [tgt_seq_len, batch_size, src_seq_len]
        output_tokens = torch.zeros(batch_size,max_length)  # [tgt_seq_len, batch_size]

        # Encoder forward pass
        # print(self.enc_embedding.weight.device, seq_x.device)

        encoder_embedded = self.enc_embedding(seq_x)
        encoder_outputs, encoder_hidden = self.enc_gru(encoder_embedded)

        # Initial decoder hidden state
        decoder_hidden = None
        # decoder_hidden = encoder_hidden
        # decoder_hidden = encoder_hidden[-1].unsqueeze(0)  # Take the last layer's hidden state
        # decoder_hidden = None

        # Start decoding
        for t in range(max_length):
            
            output_probs, decoder_hidden, attention_weights = self(seq_x, None, decoder_hidden, output_attention=True)

            # # Compute attention weights
            # attention_weights = self.attention(encoder_outputs, decoder_hidden)

            # # Store attention weights
            attentions.append(attention_weights)

            # # Compute context vector using attention weights
            # context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
            # # context = context.squeeze(1)  # [batch_size, hidden_size]

            # # Decoder input at time t
            # # decoder_input = self.dec_embedding(output_tokens[t].long().unsqueeze(1))

            # # # Concatenate context vector and decoder input
            # # decoder_input_concat = torch.cat((decoder_input, context.unsqueeze(1)), dim=-1)

            # # Decoder forward pass
            # decoder_output, decoder_hidden = self.dec_gru(context, decoder_hidden)

            # # Compute output token probabilities
            # output_probs = F.softmax(self.dec_out(decoder_output.squeeze(1)), dim=1)

            # # Choose the token with the highest probability as the output token
            output_tokens[:,t] = torch.argmax(output_probs, dim=1)

        attentions = torch.cat(attentions, dim=-2).squeeze(0)
        # print("Output before: ",output_tokens.shape)
        output_tokens = output_tokens.squeeze(0)
        # print("Output after: ",output_tokens.shape)
        return attentions.cpu(), output_tokens.cpu()

        # END CODE

    def forward(self, inputs, decoder_inputs=None, decoder_hidden_state=None, output_attention=False):
        """ Performs a forward pass over the encoder-decoder network.

            Accepts inputs for the encoder, inputs for the decoder, and hidden state for
                the decoder to continue generation after the given input.

        Args:
            inputs (torch.Tensor): tensor of shape [batch_size?, src_seq_len]
            decoder_inputs (torch.Tensor): Decoder inputs, as tensor of shape [batch_size?, 1]
            decoder_hidden_state (any): tensor to represent decoder hidden state from time step T-1.
            output_attention (bool): If true, this function should also return the
                associated attention weights for the time step, of shape [batch_size?, 1, src_seq_len].

        Returns:
            tuple[torch.Tensor, any]: output from the decoder, and associated hidden state for the next step.

            Decoder outputs should be log probabilities over the target vocabulary.

        Example:
        >>> model = RNNEncoderDecoderWithAttention(*args, **kwargs)
        >>> output, hidden = model(..., output_attention=False)
        >>> output, hidden, attn_weights = model(..., output_attention=True)
        """

        # BEGIN CODE : enc-dec-rnn-attn.forward

        if decoder_hidden_state is None:
            encoder_embedded = self.enc_embedding(inputs)
            encoder_outputs, encoder_hidden = self.enc_gru(encoder_embedded)
            decoder_hidden_state = torch.zeros(
                encoder_hidden.shape[0], encoder_hidden.shape[-2], self.hidden_size,
                device=self.device
            )
        else:
            encoder_outputs, decoder_hidden_state = decoder_hidden_state

        attn_mask = (inputs != 0)
        attention_weights = self.attention(encoder_outputs, decoder_hidden_state, attn_mask)

        context = torch.bmm(attention_weights, encoder_outputs)
        decoder_outputs, decoder_hidden_state = self.dec_gru(context, decoder_hidden_state)

        # Compute output logits
        logits = self.dec_out(decoder_outputs).squeeze(1)

        # Compute log probabilities
        logits = torch.log_softmax(logits, dim=-1)

        if output_attention:
            return logits, (encoder_outputs, decoder_hidden_state), attention_weights
        else:
            return logits, (encoder_outputs, decoder_hidden_state)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

        # END CODE

## ==== END EVALUATION PORTION

# %%
## == BEGIN EVALUATION PORTION

# Edit the hyperparameters below to your desired values.

# BEGIN CODE : rnn-enc-dec-attn.params

# Add parameters related to the model here.
rnn_enc_dec_attn_params = {
    'src_vocab_size': SRC_VOCAB_SIZE,
    'tgt_vocab_size': TGT_VOCAB_SIZE,
    'embd_dims'     : 512,
    'hidden_size'   : 512,
    'dropout'       : 0.1,
    'num_layers'    : 1
}

# Add parameters related to the dataset processing here.
rnn_enc_dec_attn_data_params = dict(
    src_padding=26,
    tgt_padding=27,
)

# Add parameters related to training here.
rnn_enc_dec_attn_training_params = dict(
    num_epochs=20,
    batch_size=32,
    shuffle=True,
    save_steps=1000,
    eval_steps=100000000
)

# END CODE

# Do not forget to set a deterministic seed.
torch.manual_seed(42)

model = RNNEncoderDecoderLMWithAttention(**rnn_enc_dec_attn_params)

# BEGIN CODE : rnn-enc-dec-attn.train

# ADD YOUR CODE HERE
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# END CODE

trainer = RNNEncoderDecoderTrainer(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec.attn"),
    model, criterion, optimizer
)

## == END EVALUATION PORTION

# %%
# Please do not change anything in the following cell.

train_dataset      = TokenizerDataset(train_data     , src_tokenizer, tgt_tokenizer, **rnn_enc_dec_attn_data_params)
validation_dataset = TokenizerDataset(validation_data, src_tokenizer, tgt_tokenizer, **rnn_enc_dec_attn_data_params)

rnn_enc_dec_attn_train_data = dict(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    collate_fn=train_dataset.collate
)
# torch.Size([2, 32, 26, 1]) torch.Size([32, 26, 512])
# Resume training from the last checkpoint, if interrupted midway, otherwise starts from scratch.
trainer.resume()
# print(trainer.last_checkpoint)

# Train as per specified training parameters.
trainer.train(**rnn_enc_dec_attn_train_data, **rnn_enc_dec_attn_training_params)

# %%
# Please do not change anything in the following cell.

# Save the final model, with additional metadata.
trainer.save(metadata={
    'model'   : rnn_enc_dec_attn_params,
    'data'    : rnn_enc_dec_attn_data_params,
    'training': rnn_enc_dec_attn_training_params
})

# %% [markdown]
# We can validate the model using a few simple tests as below:

# %%
# Please do not change anything in the following cell.

for _, row in train_data.sample(n=5, random_state=42).iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )

    print("Name                      :", row['Name'])
    print("Translation (Expected)    :", row['Translation'])
    print("Translation (Model)       :", y_pred)

    print()

# %%
# Please do not change anything in the following cell.

for _, row in validation_data.sample(n=5, random_state=42).iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )

    print("Name                      :", row['Name'])
    print("Translation (Expected)    :", row['Translation'])
    print("Translation (Model)       :", y_pred)

    print()

# %% [markdown]
# It may also be useful to look at attention maps for different examples:

# %%
# Please do not change anything in the following cell.

def visualize_attention(src_glyphs, tgt_glyphs, attention, axes):
    axes.matshow(attention.numpy(), cmap='bone')

    axes.set_xticks(numpy.arange(len(src_glyphs)), labels=src_glyphs)
    axes.set_yticks(numpy.arange(len(tgt_glyphs)), labels=tgt_glyphs)

# %%
# Please do not change anything in the following cell.

pyplot.figure(figsize=(12, 10))

src_id_to_token = inverse_vocabulary(src_tokenizer)
tgt_id_to_token = inverse_vocabulary(tgt_tokenizer)

with torch.no_grad():
    for i, row in train_data.sample(n=4, random_state=69, ignore_index=True).iterrows():
        src_tokens = torch.tensor(src_tokenizer.encode(row['Name']))
        attentions, tgt_tokens = model.attentions(src_tokens, tgt_tokenizer.get_special_tokens()['[CLS]'], max_length=50)
        src_glyphs = apply_inverse_vocab(src_tokens.tolist(), src_id_to_token)
        tgt_glyphs = apply_inverse_vocab(tgt_tokens.tolist(), tgt_id_to_token)
        axes = pyplot.subplot(2, 2, i+1)
        visualize_attention(src_glyphs, tgt_glyphs, attentions, axes)

# %%
# Please do not change anything in the following cell.

output_data = []
for _, row in validation_data.iterrows():
    y_pred = rnn_greedy_generate(
        model, row['Name'], src_tokenizer, tgt_tokenizer,
        max_length = rnn_enc_dec_data_params['tgt_padding']
    )
    output_data.append({ 'Name': row['Name'], 'Translation': y_pred })

pd.DataFrame.from_records(output_data).to_csv(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec.attn", "outputs.csv"), index=False
)

# %%
# Please do not change anything in the following cell.

# Release resources
if 'trainer' in globals():
    del trainer

if 'model' in globals():
    del model

sync_vram()

# %% [markdown]
# ## Evaluation

# %% [markdown]
# In the last few modules, you have implemented different approaches towards transliteration of Indian names to Hindi. To assess how well different systems perform, it is useful to compute different metrics, which assess different properties:
# 
# - **Accuracy**: From a parallel corpus, number of translations the model got exactly right. Higher the better. Note that this makes sense only for this task. and lacks granularity.
# - **Edit Distance**: Number of edits at the character level (insertions, deletions, substitutions) required to transform your model's outputs to a reference translation. Lower the better.
# - **Character Error Rate (CER)**: The rate at which your system/model makes mistakes at the character level. Lower the better.
# - **Token Error Rate (TER)**: The rate at which your system/model makes mistakes at the token level. Lower the better. Depending on your tokenizer implementation, could be the same as CER.
# - **BiLingual Evaluation Understudy (BLEU)**: Proposed by [Papineni et al., 2002](https://aclanthology.org/P02-1040/), BLEU is a metric that assess the quality of a translation against reference translations through assessing n-gram overlap. Higher the better.
# 
# Since accents and half-letters exist as separate characters in the Unicode specification, and can change the interpretation of the output, metrics that operate at the character level will treat these separately.

# %%
# Please do not change anything in the following cell.

class Evaluator:
    """ Class to handle all the logic concerning the evaluation of trained models.  """

    def __init__(self, src_tokenizer, tgt_tokenizer) -> None:
        """ Initializes the evaluator.

        Args:
            src_tokenizer (Tokenizer): Tokenizer for input strings in the source language.
            tgt_tokenizer (Tokenizer): Tokenizer for output strings in the target language.
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.decoding_method = None

    def set_decoding_method(self, decoding_method):
        """ Sets the decoding method to use with models.
                The evaluation function will use the set decoding method to generate outputs from the model.

        Args:
            decoding_method (function): Decoding method.
                Must accept the model instance, the input string, and tokenizers as arguments.
                Can accept additional arguments if required.
        """

        self.decoding_method = decoding_method

    @staticmethod
    def decompose(string):
        """ Decomposes a string into a set of tokens.

        Args:
            string (str): String to decompose.

        Returns:
            list[str]: List of characters from the string.
        """
        return unicodedata.normalize('NFKD', string).encode('utf-8')

    @staticmethod
    def levenshtein_distance(string1, string2):
        """ Computes the levensthein distance between two strings.

        Args:
            string1 (list[any]): Sequence A.
            string2 (list[any]): Sequence B.

        Returns:
            tuple[int, int, int]: Number of insertions + deletions, substitutions and no-ops.
        """

        costs = [
            [ 0 for j in range(len(string2)+1) ]
            for i in range(len(string1)+1)
        ]

        # Prepare matrix of costs.
        for i in range(len(string1)+1): costs[i][0] = i
        for j in range(len(string2)+1): costs[0][j] = j
        for i in range(1, len(string1)+1):
            for j in range(1, len(string2)+1):
                costs[i][j] = min(
                    costs[i][j-1] + 1,
                    costs[i-1][j] + 1,
                    costs[i-1][j-1] + (0 if string1[i-1] == string2[j-1] else 1)
                )

        # Decode matrix in backward manner for actual operation counts.
        c_ins_del, c_sub, c_noop = 0, 0, 0

        i, j = len(string1), len(string2)
        while i > 0 or j > 0:
            if i > 0 and costs[i][j] == costs[i-1][j] + 1:
                c_ins_del += 1
                i -= 1
            elif j > 0 and costs[i][j] == costs[i][j-1] + 1:
                c_ins_del += 1
                j -= 1
            elif i > 0 and j > 0:
                if string1[i-1] == string2[j-1]:
                    c_noop += 1
                else:
                    c_sub += 1
                i, j = i-1, j-1
            else:
                break

        return c_ins_del, c_sub, c_noop

    @staticmethod
    def accuracy(y_true, y_pred):
        """ Computes the accuracy of the predictions, against a reference set of predictions.

        Args:
            y_true (list[str]): Actual translations.
            y_pred (list[str]): Generated translations.

        Returns:
            float: Accuracy score, between 0 and 1.
        """
        return sum(yi_true == yi_pred for yi_true, yi_pred in zip(y_true, y_pred)) / len(y_pred)

    @classmethod
    def char_error_rate(cls, y_true, y_pred):
        """ Computes the character level error rate (CER) of the set of
            predictions against the reference translations.

        Args:
            y_true (list[str]): Actual translations.
            y_pred (list[str]): Generated translations.

        Returns:
            float: CER score, between 0 and 1. Lower the better.
        """

        cer_score = 0

        for yi_true, yi_pred in zip(y_true, y_pred):
            yi_true, yi_pred = cls.decompose(yi_true), cls.decompose(yi_pred)
            c_ins_del, c_sub, c_noop = cls.levenshtein_distance(yi_true, yi_pred)
            cer_score += (c_ins_del + c_sub) / (c_ins_del + c_sub + c_noop)

        return cer_score / len(y_true)

    def token_error_rate(self, y_true, y_pred):
        """ Computes the token level error rate (TER) of the set of
            predictions against the reference translations.

        Args:
            y_true (list[str]): Actual translations.
            y_pred (list[str]): Generated translations.

        Returns:
            float: TER score, between 0 and 1. Lower the better.
        """

        ter_score = 0

        for yi_true, yi_pred in zip(y_true, y_pred):
            yi_true = self.tgt_tokenizer.encode(yi_true, add_start=False, add_end=False)
            yi_pred = self.tgt_tokenizer.encode(yi_pred, add_start=False, add_end=False)
            t_ins_del, t_sub, t_noop = self.levenshtein_distance(yi_true, yi_pred)
            ter_score += (t_ins_del + t_sub) / (t_ins_del + t_sub + t_noop)

        return ter_score / len(y_true)

    @classmethod
    def bleu_score(cls, y_true, y_pred):
        """ Computes the average BLEU score of the set of predictions against the reference translations.

            Uses default parameters and equal weights for all n-grams, with max N = 4. (Thus computes BLEU-4).
            Uses a smoothing method for the case of missing n-grams.

        Args:
            y_true (list[str]): Actual translations.
            y_pred (list[str]): Generated translations.

        Returns:
            float: BLEU-4 score, the higher the better.
        """

        y_true = [ [ cls.decompose(yi) ] for yi in y_true ]
        y_pred = [ cls.decompose(yi) for yi in y_pred ]

        smoothing = bleu_score.SmoothingFunction()

        return bleu_score.corpus_bleu(
            y_true, y_pred,
            smoothing_function=smoothing.method1
        )

    def evaluate(self, model_path, data, reference_outputs, **decoding_kwargs):
        """ Performs the evaluation of a specified model over given data.

        Args:
            model_path (str): Path to load the model from. Must have a model.pt file.
            data (list[str]): List of input strings to translate.
            reference_outputs (list[str]): List of output strings to use as reference.
            decoding_kwargs (dict[str, any]): Additional arguments to forward to the decoding method.
                This could be for instance, max_length for a greedy decoding method.

        Raises:
            ValueError: If the decoding method is not set apriori.
        """

        if self.decoding_method is None:
            raise ValueError(f"{self.evaluate.__name__}: no decoding method is set, assign before use.")

        # Load the model to the active device.
        model = torch.load(os.path.join(model_path, 'model.pt'), map_location=self.device)

        # Set model use parameters.
        model.to(self.device)
        model.eval()

        # Generate outputs.
        generated_outputs = []
        with torch.no_grad():
            for seq_x in data:
                generated_outputs.append(self.decoding_method(
                    model, seq_x, self.src_tokenizer,
                    self.tgt_tokenizer, **decoding_kwargs
                ))

        accuracy_score = self.accuracy(reference_outputs, generated_outputs)
        cer_score      = self.char_error_rate(reference_outputs, generated_outputs)
        ter_score      = self.token_error_rate(reference_outputs, generated_outputs)
        blue_score     = self.bleu_score(reference_outputs, generated_outputs)

        print("EVALUATION:", ">", "accuracy:", f"{accuracy_score:.2%}")
        print("EVALUATION:", ">", "CER     :", f"{cer_score:.2%}")
        print("EVALUATION:", ">", "TER     :", f"{ter_score:.2%}")
        print("EVALUATION:", ">", "BLEU    :", f"{blue_score:.4f}")
        print()

        # Free resources once evaluation is complete.
        del model
        sync_vram()

# %%
# Please do not change anything in the following cell.

evaluator = Evaluator(src_tokenizer, tgt_tokenizer)

# Use greedy decoding for producing outputs.
evaluator.set_decoding_method(rnn_greedy_generate)

# Evaluate enc-dec-rnn
print("EVALUATION:", "enc-dec-rnn")
evaluator.evaluate(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec"),
    validation_data['Name'], validation_data['Translation'],
    max_length = rnn_enc_dec_data_params['tgt_padding']
)

# Evaluate enc-dec-rnn-attn
print("EVALUATION:", "enc-dec-rnn-attn")
evaluator.evaluate(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec.attn"),
    validation_data['Name'], validation_data['Translation'],
    max_length = rnn_enc_dec_attn_data_params['tgt_padding']
)

# %% [markdown]
# ## (**Bonus**) Decoding Strategies

# %% [markdown]
# A conditional language model aims to learn $P_\theta(y | x)$, that is, the probability of the target sequence being $y$ when the input sequence is $x$. This is modeled as $P_{\theta}(y | x) = \prod_{i=1}^{|y|} {P_\theta(y_i | x, y_{1:i-1})}$.
# 
# For translation, our goal is to find the sequence that maximizes this conditional probability, i.e. $y^* = \arg \max_{y} P_\theta(y | x)$. $y^*$ is then the 'best' translation for the input sequence $x$. However, computing probabilities for all possible $y$ to find the maximizer is intractable. As a result, decoding strategies are employed to produce reasonable approximations of $y^*$.
# 
# In the last module, you evaluated your models through different metrics, but the approach for generating outputs from the model was fixed to greedy decoding, where at each time step, the token to be produced is determined by $y_{i,greedy} := \arg \max_{y_i} P(y_i| x, y_{1:i-1})$. While this approach is fast, $P(y_{greedy}|x)$ may be much less than $P(y^*|x)$. Fortunately, better decoding strategies exist to produce better approximations, however at the cost of higher time complexity. Some of these are:
# 
# - **Beam-Search Decoding**: At every time step, retains $k$ candidate token generations, which are decoded individually (each path is referred as a beam) to obtain $k$ successors per beam. For the next time step, the best $k$ candidates are retained such that conditional probability of the sequence generated so far is maximized. Has a complexity of $O(kV|y|)$, where $V$ is the size of the target vocabulary, and $|y|$ is the target sequence length. Using $k=1$ makes it equivalent to greedy decoding. Implementations also employ length penalties to not be biased towards larger target sequences.
# 
# - **Viterbi Decoding**: A decoding technique based on the Viterbi algorithm, which is a dynamic programming algorithm that provides an efficient way of finding the "most likely state sequence in the maximum a posteriori probability sense of a process assumed to be a finite-state discrete-time Markov process". Works well under assumptions of a small target vocabulary size and conditional independence of feature vectors. The decoded sequence has the minimum error probability possible when compared to $y^*$, i.e., $P(y_{viterbi}|x) \sim P(y^* | x)$. Has a time complexity of $O(V^2|y|)$.
# 
# In the next cell, you will implement any preferred decoding strategy of your choice, and compare performances of different decoding strategies.

# %% [markdown]
# **Note**: This module is optional, and can be attempted for a bonus score. If you choose to attempt this module, please set the variable `ATTEMPTED_BONUS` (declared below) to `True`, as this will impact the evaluation of the bonus part.

# %%
## ==== BEGIN EVALUATION PORTION

# ADD YOUR CODE HERE
# Set this variable to True if you choose to implement the function `rnn_better_generate` for the bonus module.
# Regardless of your choice, please run this cell.
ATTEMPTED_BONUS = True

## ==== END EVALUATION PORTION

print("EVALUATION:", "Attempted Bonus Module?", ATTEMPTED_BONUS)

# %%
## ==== BEGIN EVALUATION PORTION

# Feel free to add additional parameters to rnn_better_generate, such as k for Beam Search Decoding.
def rnn_better_generate(model, seq_x, src_tokenizer, tgt_tokenizer, max_length, k=5, length_penalty_alpha=0.6):
    """ Given a source string, translate it to the target language using the trained model.
        This function should use a better decoding strategy than greedy decoding (see above) to generate the results.

    Args:
        model (nn.Module): RNN Type Encoder-Decoder Model
        seq_x (str): Input string to translate.
        src_tokenizer (Tokenizer): Source language tokenizer.
        tgt_tokenizer (Tokenizer): Target language tokenizer.
        max_length (int): Maximum length of the target sequence to decode.

    Returns:
        str: Generated string for the given input in the target language.
    """

    # BEGIN CODE : enc-dec-rnn.better_generate

    # ADD YOUR CODE HERE
    src_tokens = src_tokenizer.encode(seq_x, add_start=True, add_end=True)
    src_tokens = torch.tensor(src_tokens).unsqueeze(0)  # Add batch dimension

    # Initialize the beam search
    beam = [(src_tokens, [], 0)]  # (current_tokens, current_output_tokens, cumulative_log_prob)

    # Initialize storage for completed sequences
    completed_seqs = []

    # Start decoding
    for _ in range(max_length):
        next_beam = []

        # Expand each beam
        for current_tokens, current_output_tokens, cumulative_log_prob in beam:
            # Perform forward pass to get logits
            logits, _ = model(current_tokens, decoder_inputs=torch.tensor(current_output_tokens))

            # Get top k predictions
            topk_probs, topk_indices = torch.topk(logits, max_length, dim=1)

            for i in range(max_length):
                token_idx = topk_indices[0][i].item()
                token_prob = topk_probs[0][i].item()

                if token_idx == tgt_tokenizer.get_special_tokens()['[SEP]']:
                    # Reached end of sequence token
                    completed_seqs.append((current_output_tokens + [token_idx], cumulative_log_prob + token_prob))
                else:
                    next_beam.append((
                        current_tokens,
                        current_output_tokens + [token_idx],
                        cumulative_log_prob + token_prob
                    ))

        # Sort the next beam by cumulative log probability
        next_beam.sort(key=lambda x: x[2], reverse=True)

        # Select top beam_width sequences
        beam = next_beam[:max_length]

    # Select the best completed sequence based on cumulative log probability
    completed_seqs.sort(key=lambda x: x[1], reverse=True)
    best_seq_tokens = completed_seqs[0][0]

    # Decode token sequence to text
    tgt_seq = tgt_tokenizer.decode(best_seq_tokens, strip_special=True)

    return tgt_seq

    # END CODE

## ==== END EVALUATION PORTION

# %%
## ==== BEGIN EVALUATION PORTION

# BEGIN CODE : decoding.init

# Add parameter values for your decoding strategy here. Leave empty if unused.

decoding_params = dict(
    k=5,  # Beam width for beam search decoding
    length_penalty_alpha=0.6  # Length penalty parameter for beam search decoding
)
## ==== END EVALUATION PORTION

# %%
# Please do not change anything in the following cell.

evaluator = Evaluator(src_tokenizer, tgt_tokenizer)

# Use a different decoding for producing outputs.
if ATTEMPTED_BONUS:
    evaluator.set_decoding_method(rnn_better_generate)
else:
    evaluator.set_decoding_method(rnn_greedy_generate)

# Evaluate enc-dec-rnn
print("EVALUATION:", "enc-dec-rnn")
evaluator.evaluate(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec"),
    validation_data['Name'], validation_data['Translation'],
    max_length = rnn_enc_dec_data_params['tgt_padding'],
    **decoding_params
)

# Evaluate enc-dec-rnn-attn
print("EVALUATION:", "enc-dec-rnn-attn")
evaluator.evaluate(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec.attn"),
    validation_data['Name'], validation_data['Translation'],
    max_length = rnn_enc_dec_attn_data_params['tgt_padding'],
    **decoding_params
)

# %% [markdown]
# ## Congratulations!

# %% [markdown]
# You have reached the end of the last assignment! Hope this was a fun exercise!
# 
# Once you've run the cells above, you should be ready to submit the assignment.

# %%



