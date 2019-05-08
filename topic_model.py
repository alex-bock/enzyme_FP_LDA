
# topic_model.py
#
# Author:       Alex Bock

# imports

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel

# TopicModel

class TopicModel:

    # class constructor ========================================================

    def __init__ (self, mallet_path=None):

        if mallet_path:
            self.mallet = True
            self.mallet_path = mallet_path
        else:
            self.mallet = False

        return

    # public methods ===========================================================

    # build_dictionary
    #
    # inputs:   list of documents as word lists (preprocessing assumed)
    # outputs:  none
    # purpose:  builds a Gensim dictionary and corpus of training set

    def build_dictionary (self, inputs):

        dictionary = Dictionary(inputs)
        corpus = [None] * len(inputs)

        for i in range(0, len(inputs)):
            corpus[i] = dictionary.doc2bow(inputs[i], allow_update=True)

        self.dictionary = dictionary
        self.corpus = corpus

        return

    # build_model
    #
    # inputs:   number of topics
    # outputs:  none
    # purpose:  constructs a Gensim LDA topic model from dictionary

    def build_model (self, n_topics):

        if n_topics == 0:
            raise Exception("Number of topics must be greater than 0")

        if self.mallet:
            model = LdaMallet(self.mallet_path,
                              corpus=self.corpus,
                              id2word=self.dictionary,
                              num_topics=n_topics)
        else:
            model = LdaModel(corpus=self.corpus,
                             id2word=self.dictionary,
                             num_topics=n_topics,
                             minimum_probability=0.0)

        self.model = model
        self.n_topics = n_topics

        return

    # analyze_model
    #
    # inputs:   none
    # outputs:  perplexity and coherence scores
    # purpose:  generates and returns coherence and perplexity statistics for
    #           generated LDA topic model

    def analyze_model (self):

        if not self.model:
            raise Exception("No model to analyze!")

        perplexity = self.model.log_perplexity(self.corpus)

        coherence_model = CoherenceModel(model=self.model,
                                         corpus=self.corpus,
                                         coherence="u_mass")
        coherence = coherence_model.get_coherence()

        return (perplexity, coherence)

    # get_topic_vector
    #
    # inputs:   document as word list (preprocessing assumed), decimal place to
    #           round to (optional, default = 3)
    # outputs:  topic vector of input document
    # purpose:  generates and returns probability distribution of topics for
    #           given word list

    def get_topic_vector (self, input, spec=3):

        if not self.model:
            raise Exception("No model for input!")

        bow = self.dictionary.doc2bow(input)
        vector = [round(pair[1], spec) for pair in self.model[bow]]

        return vector
    
    # get_topics
    #
    # inputs:   none
    # outputs:  distribution of generated topics
    # purpose:  returns probability-constituent distributions for each
    #           generated topic

    def get_topics (self):

        return self.model.print_topics(self.n_topics)

    # get_num_topics
    #
    # inputs:   none
    # outputs:  number of topics
    # purpose:  public getter for n_topics

    def get_num_topics (self):

        return self.n_topics

    # is_mallet
    #
    # inputs:   none
    # outputs:  whether model is a MALLET model
    # purpose:  public getter for mallet

    def is_mallet (self):

        return self.mallet
