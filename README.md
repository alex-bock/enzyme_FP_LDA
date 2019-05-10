# enzyme_FP_LDA

This is a continuing project aimed at using topic modeling to generate computationally compatible fingerprint representations of amino acid sequences. Only the code used to simulate topic modeling on amino acid sequences is available here due to the size of the datasets I've used to build/test this code.

## Prerequisites

To successfully run this code, you will need:

* Gensim (https://radimrehurek.com/gensim/)
* Python 2.7 (behavior with python3 is undefined as the code has (so far) been written and tested using Python 2.7)
* Python packages argparse and pickle
* A directory to which the code will write generated data
* In that directory, a .csv file containing amino acid sequences in the first column and corresponding Enzyme Commission (EC) identifiers in the second

## Building a topic model

The program build_model.py builds a topic model with a specified number of topics on a speficied amino acid dataset. If your dataset is large and you wish to train on a subset of it, there's currently a variable in the generate_ngrams() call in build_model.py that allows you to specify the number of sequences on which to train the topic model.

This program requires the following command-line arguments:

* -d path/to/sequence/data (directory level only)
* -n \[number of topics (positive nonzero integer)\]

Upon building a topic model, the program will report coherence and perplexity evaluation statistics and write the topic model to file in path/to/sequence/data. The filename will be of the form "LDA_\[number of topics\]t_gensim.sav."

## Using a topic model

The program test_model.py enables you to use an existing, trained topic model that has been written to file to generate topic distributions for an amino acid dataset. 

This program requires the following command-line arguments:

* -m \[path/to/topic_model/file.sav]\
* -ec \[EC taxonomy to be used as testing set; a string of the form "x.x.x.x"\]
* -a (optional; runs coherence and perplexity analysis of topic model before running on testing set)
* -d path/to/sequence/data (directory level only)

This program generates a topic distribution for each sequence in the specified EC taxonomy and writes these distributions to a .csv file in path/to/sequence/data. The filename will be of the form "\[EC taxonomy\]\_\[number of topics\]t_gensim.csv."
