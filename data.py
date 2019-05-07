
# data.py
#
# Author:       Alex Bock

# imports

import csv

# Data

class Data:

    # class constructor ========================================================

    # __init__
    #
    # inputs:   name of sequence data file
    # outputs:  none
    # purpose:  class constructor; reads a sequence data file and compiles
    #           sequences and corresponding EC numbers into data structures

    def __init__ (self, filename):

        file = open(filename, "r")
        reader = csv.reader(file, delimiter=',')
        data = list(reader)

        self.n_seqs = len(data)
        self.seqs = [""] * self.n_seqs
        self.ecs = [()] * self.n_seqs

        for i in range(0, self.n_seqs):
            seq = data[i][0].decode("utf-8-sig")
            ec = data[i][1].decode("utf-8-sig")
            self.seqs[i] = seq
            self.ecs[i] = ec

    # private methods ==========================================================

    # __filter_by_EC
    #
    # inputs:   EC identifier
    # outputs:  list of sequences
    # purpose:  returns all sequences in the dataset sharing a given EC
    #           identifier

    def __filter_by_EC (self, ec):

        ec = ec.split('.')
        spec = len(ec)
        print("Specificity: {}".format(spec))

        seqs = []

        for i in range(0, self.n_seqs):
            this_ec = self.ecs[i].split('.')
            if ec == this_ec[:spec]:
                seqs.append(self.seqs[i])

        return seqs

    # __ngrams
    #
    # inputs:   sequence to be fragmented into n-grams, n value
    # outputs:  list of n-grams of sequence
    # purpose:  generates an n-gram representation of the provided sequence

    def __ngrams (self, seq, n):

        n_ngrams = len(seq) - (n - 1)
        ngrams = [""] * n_ngrams

        for i in range(0, n_ngrams):
            ngram = seq[i:(i+n)]
            ngrams[i] = ngram

        return ngrams

    # __generate_ngrams
    #
    # inputs:   n value
    # outputs:  list of n-gram representations
    # purpose:  generates and returns n-gram representations of every sequence
    #           in the provided sequence list

    def __generate_ngrams (self, seqs, n):

        n_seqs = len(seqs)
        ngrams_list = [[]] * n_seqs

        for i in range(0, n_seqs):
            ngrams = self.__ngrams(seqs[i], n)
            ngrams_list[i] = ngrams

        return ngrams_list

    # public methods ===========================================================

    # generate_ngrams
    #
    # inputs:   n value, EC identifier (optional), number of sequences to
    #           consider
    # outputs:  list of n-gram representations
    # purpose:  returns n-gram representations of all sequences in defined
    #           (sub)set; can be entire dataset (default) or sequences with a
    #           particular EC identifier (if a value is provided for ec)

    def generate_ngrams (self, n, ec=None, limit=-1):

        if ec:
            seqs = self.__filter_by_EC(ec)
        else:
            seqs = self.seqs
            if limit != -1:
                seqs = seqs[:limit]

        ngrams_list = self.__generate_ngrams(seqs, n)

        return ngrams_list

    # get_ngrams_by_index
    #
    # inputs:   index, n value
    # outputs:  n-gram representations
    # purpose:  generates and returns n-gram representation of the ith sequence
    #           in the dataset; raises an exception if i exceeds index of last
    #           sequence

    def get_ngrams_by_index (self, i, n):

        if i >= self.n_seqs:
            raise Exception("Sequence {} requested but only {} sequences exist"\
                            .format(i, self.n_seqs))

        seq = self.seqs[i]
        ngrams = self.__ngrams(seq, n)

        return ngrams
