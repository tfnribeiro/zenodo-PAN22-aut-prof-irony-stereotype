from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from read_files import X,y
from read_files import USERCODE_X
from utils import tokenize_tweet
import numpy as np
import regex as re
from io import StringIO
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def collect(array):

    train_author = [None for i in range(array.shape[0])]

    for i, author in enumerate(array):
        buf = StringIO()
        for tweet in author:
            buf.write(tweet + "")
        single_text = buf.getvalue()
        train_author[i] = single_text

    train_author = np.array(train_author)
    return train_author 

def baseline_svm(data, labels, tokenize = 'char', kfold = 5):

    print("Tokenizing and vectorizing input...")

    if tokenize == 'word':
        tknzr = TweetTokenizer()
        vectorizer = CountVectorizer(analyzer='word', tokenizer = tknzr.tokenize, ngram_range=(2, 2))
        X = vectorizer.fit_transform(collect(data))
    else:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        X = vectorizer.fit_transform(collect(data))

    accuracies = []

    print("Initializing K-fold splits...")

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        i= 0
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = svm.SVC()
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        print(f"Split {i} Accuracy: {accuracy_score(y_test, prediction)}")
        accuracies.append(prediction)
        i+= 1

    print(type(accuracies), type(accuracies[0]))
    print("Average accuracy:", sum(accuracies)/kfold)






