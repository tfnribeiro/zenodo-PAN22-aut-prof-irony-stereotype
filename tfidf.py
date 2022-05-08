import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf:
    def __init__(self, corpus, terms_filter=None, lowercase=False, authors_document=False) -> None:
        """
            terms_filter = set() of reduced words, can be emojis for istance.
        """
        self.lowercase = lowercase
        if self.lowercase:
            corpus = np.char.lower(corpus)
        # Get terms in corpus:
        # Number of terms found
        self.idf_terms = dict()
        # Documents == N of Tweets
        if authors_document:
            self.n_documents = len(corpus)
        else:
            self.n_documents = 0
        self.n_terms = 0
        for author in corpus:
            if not authors_document:
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

        self.idf_matrix = np.zeros((self.n_documents, len(self.idf_terms.keys())),dtype=bool)

        document_i = 0
        for author in corpus:
            for tweet in author:
                for w in set(tokenize_tweet(tweet, as_list=True)):
                    if w in self.idf_terms:
                        self.idf_matrix[document_i, self.idf_terms[w]] = True
                if not authors_document:
                    document_i += 1
            if authors_document:
                document_i += 1
        
        # 1 x d
        self.term_idf = np.log(self.n_documents / (1+np.sum(self.idf_matrix, axis=0)))

    def tf(self, author):
        if self.lowercase:
            author = np.char.lower(author)
        tf_matrix = np.zeros((len(author), len(self.idf_terms.keys())), dtype="int16")
        for doc_i, tweet in enumerate(author):
            for w in tokenize_tweet(tweet, as_list=True):
                if w in self.idf_terms:
                    index = self.idf_terms[w]
                    tf_matrix[doc_i, index] += 1
                else:
                    continue
            tf_matrix[doc_i, :]/(tf_matrix[doc_i,:].sum()+1)
        return tf_matrix
    
    def tf_idf(self, author):
        if self.lowercase:
            author = np.char.lower(author)
        tf_vector = self.tf(author)
        return (tf_vector * self.term_idf).mean(axis=0)
