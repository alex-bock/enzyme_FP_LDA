
# topic_model.py
#
# Author:       Alex Bock

# imports

from data import Data
from topic_model import TopicModel

import csv

ngram_length = 3
n_seqs = 10000

print("Generating ngrams...")
d = Data("sequence_data.csv")
ngrams = d.generate_ngrams(ngram_length, limit=n_seqs)

print("Building dictionary...")
LDA = TopicModel()
LDA.build_dictionary(ngrams)

print("Running sequential topic number analysis...")

output = csv.writer(open("model_analysis.csv", "w"), delimiter=",")

for n_topics in range(0, 100):
    print("{} topics".format(n_topics + 1))
    LDA.build_model(n_topics + 1)
    (perplexity, coherence) = LDA.analyze_model()
    output.writerow([n_topics, perplexity, coherence])
    LDA.delete_model()
