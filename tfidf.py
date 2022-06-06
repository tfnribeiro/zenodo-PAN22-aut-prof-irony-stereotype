import numpy as np
from tempfile import mkdtemp
import os.path as path
from utils import *

class tfidf:
    def __init__(self, corpus, terms_filter=None, lowercase=False, authors_document=False) -> None:
        """
            corpus = List of Lists with tweets
            terms_filter = set() of reduced words, can be emojis for istance.
            lowercase (bool) sets the tweets to lowercase
            authors_document uses each author as a document
        """
        self.lowercase = lowercase
        if self.lowercase:
            corpus = np.char.lower(corpus)
        # Get terms in corpus:
        # Number of terms found
        self.idf_terms = dict()
        
        self.authors_document = authors_document
        print("Generating IDF Vector")
        if self.authors_document:
            # Documents == N of Authors
            self.n_documents = len(corpus)
        else:
            # Documents == N of Tweets
            self.n_documents = 0
        self.n_terms = 0
        for author in corpus:
            if not self.authors_document:
                self.n_documents += len(author)
            for tweet in author:
                for w in set(tokenize_tweet(tweet, as_list=True)):
                    if terms_filter:
                        if w not in self.idf_terms and w in terms_filter:
                            self.idf_terms[w] = self.n_terms
                            self.n_terms += 1
                    else:
                        if w not in self.idf_terms:
                            self.idf_terms[w] = self.n_terms
                            self.n_terms += 1

        self.index_to_term = {v:k for k,v in self.idf_terms.items()}
        try:
            self.idf_matrix = np.zeros((self.n_documents, len(self.idf_terms.keys())),dtype=bool)
        except MemoryError:
            print("WARNING: Not enough memory to create idf matrix, using disk.")
            mem_file = path.join(mkdtemp(), 'tf_idf.dat')
            self.idf_matrix = np.memmap(mem_file, dtype='bool', mode='w+', shape=(self.n_documents, len(self.idf_terms.keys())))

        document_i = 0
        for author in corpus:
            for tweet in author:
                for w in set(tokenize_tweet(tweet, as_list=True)):
                    if w in self.idf_terms:
                        self.idf_matrix[document_i, self.idf_terms[w]] = True
                if not self.authors_document:
                    document_i += 1
            if self.authors_document:
                document_i += 1
        
        # 1 x d, term_idf vector
        self.term_idf = np.log2(self.n_documents / (np.sum(self.idf_matrix, axis=0)))

    def tf(self, author):
        """
            Calculate term frequency for a specific author.
            Returns the TF count matrix.
        """
        if self.lowercase:
            author = np.char.lower(author)
        if self.authors_document:
            tf_matrix = np.zeros((1, len(self.idf_terms.keys())), dtype="int16")
            for tweet in author:
                for w in tokenize_tweet(tweet, as_list=True):
                    if w in self.idf_terms:
                        index = self.idf_terms[w]
                        tf_matrix[0, index] += 1
                    else:
                        continue
        else:
            tf_matrix = np.zeros((len(author), len(self.idf_terms.keys())), dtype="int16")
            for doc_i, tweet in enumerate(author):
                for w in tokenize_tweet(tweet, as_list=True):
                    if w in self.idf_terms:
                        index = self.idf_terms[w]
                        tf_matrix[doc_i, index] += 1
                    else:
                        continue
        
        tf_matrix[tf_matrix>0] = 1 + np.log2(tf_matrix[tf_matrix>0])
        return tf_matrix
    
    def tf_idf(self, author):
        """
            Uses the TF count matrix to generate the average TF-IDF vector.
        """
        if self.lowercase:
            author = np.char.lower(author)
        tf_vector = self.tf(author)
        tf_vector = tf_vector * self.term_idf
        average_tf_vector = (tf_vector).mean(axis=0)
        return average_tf_vector
