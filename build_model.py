
# build_model.py
#
# Author:       Alex Bock

# imports

from data import Data
from topic_model import TopicModel

import argparse
from pickle import dump
import sys

# parse command line arguments and set variables

parser = argparse.ArgumentParser()

parser.add_argument("-d", action="store", dest="data_location")
parser.add_argument("-n", action="store", dest="n_topics")
parser.add_argument("-mallet", action="store_true", dest="mallet")
parser.add_argument("-mp", action="store", dest="mallet_path")

arguments = parser.parse_args()
data_location = arguments.data_location
n_topics = arguments.n_topics
mallet = arguments.mallet
mallet_path = arguments.mallet_path

ngram_length = 3
n_seqs = 10000          # get first n sequences from dataset

# Step 1: Generate ngrams of training sequence data (first 10K sequences)

print("Generating ngrams for training set...")

d = Data(data_location + "sequence_data.csv")
training_set = d.generate_ngrams(ngram_length, limit=n_seqs)

# Step 2: Build Gensim LDA topic model on training sequence data

LDA_file = data_location + "LDA_" + str(n_topics) + "t_" +\
           ("mallet" if mallet else "gensim") + ".sav"

try:
    infile = open(LDA_file, "rb")
except IOError:
    infile = None

if infile:
    sys.exit("Model already exists!")
else:
    if mallet:
        LDA = TopicModel(mallet_path=mallet_path)
    else:
        LDA = TopicModel()
    print("Building dictionary...")
    LDA.build_dictionary(training_set)
    print("Building model...")
    LDA.build_model(n_topics)
    print("Saving model to file...")
    dump(LDA, open(LDA_file, "wb"))

# Step 3: Get performance statistics for trained model

print("Analyzing model...")

(perplexity, coherence) = LDA.analyze_model()
print("Perplexity:  {}\nCoherence:  {}".format(perplexity, coherence))
