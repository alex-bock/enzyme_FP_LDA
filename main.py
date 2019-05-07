
# main.py
#
# Author:       Alex Bock

# imports

from data import Data
from topic_model import TopicModel

import csv
from pickle import load, dump

ngram_length = 3
n_seqs = 10000
n_topics = 20
ec = "1.14.12"

# Step 1: Generate ngrams of training sequence data (first 10K sequences)

print("Generating ngrams for training set...")

d = Data("data/sequence_data.csv")
training_set = d.generate_ngrams(ngram_length, limit=n_seqs)

# Step 2: Build Gensim LDA topic model on training sequence data

print("Building dictionary and model...")

try:
    infile = open("data/LDA.sav", "rb")
except IOError:
    infile = None

if infile:
    print("Reading model from file...")
    LDA = load(infile)
else:
    LDA = TopicModel()
    LDA.build_dictionary(training_set)
    LDA.build_model(n_topics)
    dump(LDA, open("data/LDA.sav", "wb"))

# Step 3: Get performance statistics for trained model

print("Analyzing model...")

(perplexity, coherence) = LDA.analyze_model()
print("Perplexity:  {}\nCoherence:  {}".format(perplexity, coherence))

# Step 4: Generate ngrams of testing sequence data (1.14.12.-)

print("Generating ngrams for testing set...")

testing_set = d.generate_ngrams(ngram_length, ec=ec)

# Step 5: Get topic vectors for each testing sequence

print("Getting topic vectors...")

vectors = [[]] * len(testing_set)

for i in range(0, len(testing_set)):
    vector = LDA.get_topic_vector(testing_set[i])
    vectors[i] = vector

# Step 6: Record topic vectors for testing sequences

print("Writing results to file...")

results = open("data/" + ec + ".csv", "wb")
writer = csv.writer(results)

for vector in vectors:
    writer.writerow(vector)

close(results)
