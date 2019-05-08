
# test_model.py
#
# Author:       Alex Bock

# imports

from data import Data

import argparse
import csv
from pickle import load

# parse command line arguments and set variables

parser = argparse.ArgumentParser()

parser.add_argument("-m", action="store", dest="LDA_file")
parser.add_argument("-ec", action="store", dest="ec")
parser.add_argument("-a", action="store_true", dest="analyze")
parser.add_argument("-d", action="store", dest="data_location")

arguments = parser.parse_args()
LDA_file = arguments.LDA_file
ec = arguments.ec
analyze = arguments.analyze
data_location = arguments.data_location

ngram_length = 3

# check if LDA file exists and load to LDA object

try:
    infile = open(LDA_file, "rb")
except IOError:
    infile = None

if infile:
    print("Loading model...")
    LDA = load(infile)
else:
    raise Exception("{} does not exist!".format(LDA_file))

# analyze model (optional)

if analyze:
    print("Analyzing model...")
    (perplexity, coherence) = LDA.analyze_model()
    print("Perplexity:  {}\nCoherence:  {}".format(perplexity, coherence))

# generate ngrams of testing sequence data (1.14.12.-)

print("Generating ngrams for testing set...")

d = Data(data_location + "sequence_data.csv")
testing_set = d.generate_ngrams(ngram_length, ec=ec)

# get topic vectors for each testing sequence

print("Getting topic vectors...")

vectors = [[]] * len(testing_set)

for i in range(0, len(testing_set)):
    vector = LDA.get_topic_vector(testing_set[i])
    vectors[i] = vector

# record topic vectors for testing sequences to file

print("Writing results to file...")

n_topics = LDA.get_num_topics()
mallet = LDA.is_mallet()

results_file = data_location + ec + "_" + str(n_topics) + "t_" +\
               ("mallet" if mallet else "gensim") + ".csv"
results = open(results_file, "wb")
writer = csv.writer(results)

for vector in vectors:
    writer.writerow(vector)
