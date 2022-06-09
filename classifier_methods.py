from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn import svm
from pos_counts import *
from count_features import *
from lexical_comp import *
# Comment out for word embeddings
# from word_emb import * 
from sent_polarity import *
from punctuation import *
from tqdm import tqdm
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os


def get_features(dataset, function, label="", verbose=False):
    """
        Generate the features based on a function to transform the original dataset.
        Inputs:
            dataset(List of Tweets)
            function(fn): Function like the ones present in count_features.py
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            np.array with the features returned by the function
    """
    list_features = []
    if verbose:
        for i in range(len(dataset)):
            tweet_list = dataset[i]
            get_features = function(tweet_list)
            list_features.append(get_features)
    else:
        print(f"Processing features: {function.__name__}")
        for i in tqdm(range(len(dataset))):
            tweet_list = dataset[i]
            get_features = function(tweet_list)
            list_features.append(get_features)
        print(f"{label} Processing for features: {function.__name__}, is complete!")
    return np.array(list_features)


def get_features_test(author_list_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, label="Generating Test Features", verbose=False):
    """
        Generate a test vector which can be provided to a trained classifier
        Inputs:
            author_list_test - list of list of strings: Author List with Tweets to be transformed into an input vector 
            emoji_pca, profanity_pca, word_pca: pca objects trained on the training data, the can be obtained when running get_features_train
            emoji_tfidf, profanity_tfidf, words_tfidf: tfidf objects trained on the trainind data, these can be obtained when running get_features_train
            NOTE: for easy use, you can use the unpacking feature of python:
                if pca_tfidf_settings = (emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)
                - You can call get_features_test(author_list_test, *pca_tfidf_settings, ...)
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            np.array of size N_authors x Feature Size. This vector can be used in a train classifier to make predictions
    """
    print(label)
    pos_features = get_features(author_list_test, pos_counts,
                                "Individual Predict", verbose=verbose)
    count_features = get_features(author_list_test, author_style_counts,
                                  "Individual Predict", verbose=verbose)
    lix_features = get_features(author_list_test, lix_score,
                                "Individual Predict", verbose=verbose)
    sent_features = get_features(author_list_test, get_sent_polarity,
                                 "Individual Predict", verbose=verbose)
    sep_punct_features = get_features(
        author_list_test, seperated_punctuation, "Individual Predict", verbose=verbose)
    
    # embedding_features = get_features(author_list_test, tweet_word_embs, "Individual Predict", verbose=verbose)
    # miss_features = get_features(test, misspelled, "Individual Predict", verbose=True).reshape((-1,1))
    emoji_tfidf_features = get_features(
        author_list_test, emoji_tfidf.tf_idf, "Individual Predict", verbose=verbose)
    profanity_tfidf_features = get_features(
        author_list_test, profanity_tfidf.tf_idf, "Individual Predict", verbose=verbose)
    words_tfidf_features = get_features(
        author_list_test, words_tfidf.tf_idf, "Words TF_IDF", verbose=verbose)
    emoji_features_test = emoji_pca.transform(emoji_tfidf_features)
    profanity_features_test = profanity_pca.transform(profanity_tfidf_features)
    word_features_test = word_pca.transform(words_tfidf_features)

    x_test = np.concatenate((pos_features, count_features, sent_features, sep_punct_features,
                            lix_features, emoji_features_test, profanity_features_test, word_features_test), axis=1)

    return x_test

def get_features_test_pca(author_list_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, label="Generating Test Features", verbose=False):
    """
        Generate PCA features, this is mostly used in Cross Validation, to ensure no data is used when training the TF-IDF/PCA objects
        Inputs:
            author_list_test - list of list of strings: Author List with Tweets to be transformed into an input vector 
            emoji_pca, profanity_pca, word_pca: pca objects trained on the training data, the can be obtained when running get_features_train
            emoji_tfidf, profanity_tfidf, words_tfidf: tfidf objects trained on the trainind data, these can be obtained when running get_features_train
            NOTE: for easy use, you can use the unpacking feature of python:
                if pca_tfidf_settings = (emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)
                - You can call get_features_test(author_list_test, *pca_tfidf_settings, ...)
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            emoji_features_test: np.array, size N authors x Emoji PCA size
            profanity_features_test: np.array, size N authors x profanity PCA size
            word_features_test: np.array, size N authors x word PCA size
    """
    print(label)
    emoji_tfidf_features = get_features(
        author_list_test, emoji_tfidf.tf_idf, "Individual Predict", verbose=verbose)
    profanity_tfidf_features = get_features(
        author_list_test, profanity_tfidf.tf_idf, "Individual Predict", verbose=verbose)
    words_tfidf_features = get_features(
        author_list_test, words_tfidf.tf_idf, "Words TF_IDF", verbose=verbose)
    emoji_features_test = emoji_pca.transform(emoji_tfidf_features)
    profanity_features_test = profanity_pca.transform(profanity_tfidf_features)
    word_features_test = word_pca.transform(words_tfidf_features)

    return emoji_features_test, profanity_features_test, word_features_test


def get_features_no_pca(author_list_train, label="Generating Train Features", verbose=False):
    """
        Generate features which do not require to be fitted. These are Features based on counts or existing models
        Inputs:
            author_list_test - list of list of strings: Author List with Tweets to be transformed into an input vector 
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            pos_features: np.array, size N authors x POS vector (pos_counts.py)
            count_features: np.array, size N authors x Count vector (count_features.py)
            lix_features: np.array, size N authors x LixScore (lexical_comp.py)
            sent_features: np.array, size N authors x VaderSentSTD(across tweets) (sent_polarity.py)
            sep_punct_features: np.array, size N authors x Punctuation Vector (punctuation.py)
    """
    print(label)
    pos_features = get_features(author_list_train, pos_counts,
                                "Individual Predict", verbose=verbose)
    count_features = get_features(author_list_train, author_style_counts,
                                  "Individual Predict", verbose=verbose)
    lix_features = get_features(author_list_train, lix_score,
                                "Individual Predict", verbose=verbose)
    sent_features = get_features(author_list_train, get_sent_polarity,
                                 "Individual Predict", verbose=verbose)
    sep_punct_features = get_features(
        author_list_train, seperated_punctuation, "Individual Predict", verbose=verbose)

    #embedding_features = get_features(author_list_train, tweet_word_embs, "Individual Predict", verbose=verbose)
    

    return pos_features, count_features, lix_features, sent_features, sep_punct_features


def get_features_train_pca(author_list_train, emoji_pca_dim=4, profanity_pca_dim=14, word_pca_dim=20, label="Generating Train Features", verbose=False):
    """
        Generate PCA and TF-IDF objects and returns the transformed data for each of these objects.
        Inputs:
            author_list_test - list of list of strings: Author List with Tweets to be transformed into an input vector
            emoji_pca_dim (int): Number of dimensions for the emoji pca: default=4
            profanity_pca_dim (int): Number of dimensions for the profanity pca: default=14
            word_pca_dim (int): Number of dimensions for the word pca: default=20
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            emoji_features_train : np.array NAuthors X Emoji_pca_dim, tranformed vector for emoji pca
            profanity_features_train : np.array NAuthors X Profanity_pca_dim, tranformed vector for profanity pca 
            word_features_train : np.array NAuthors X Word_pca_dim, tranformed vector for word pca
            emoji_pca : Emoji PCA object (sklearn)
            profanity_pca : Profanity PCA object (sklearn)
            word_pca : Word PCA object (sklearn)
            emoji_tfidf : Emoji TFIDF object (tfidf.py)
            profanity_tfidf : Profanity TFIDF object (tfidf.py)
            words_tfidf : Word TFIDF object (tfidf.py)
    """
    print(label)
    #miss_features = get_features(train, misspelled, "Individual Predict", verbose=True).reshape((-1,1))
    #emoji_features = get_features(train, emoji_embeds, "Individual Predict", verbose=True)
    #profanity_features = get_features(train, profanity_embeds, "Individual Predict", verbose=True)
    emoji_pca_n = emoji_pca_dim
    profanity_pca_n = profanity_pca_dim
    word_pca_n = word_pca_dim

    emoji_pca = PCA(n_components=emoji_pca_n)
    profanity_pca = PCA(n_components=profanity_pca_n)
    word_pca = PCA(n_components=word_pca_n)

    emoji_tfidf = fit_emoji_embeds_tfidf(
        author_list_train, authors_document=False)
    emoji_tfidf_features = get_features(
        author_list_train, emoji_tfidf.tf_idf, "Emoji TF_IDF", verbose=verbose)

    profanity_tfidf = fit_profanity_embeds_tfidf(
        author_list_train, authors_document=False)
    profanity_tfidf_features = get_features(
        author_list_train, profanity_tfidf.tf_idf, "Profanity TF_IDF", verbose=verbose)

    words_tfidf = fit_word_embeds_tfidf(
        author_list_train, authors_document=False)
    words_tfidf_features = get_features(
        author_list_train, words_tfidf.tf_idf, "Words TF_IDF", verbose=verbose)

    emoji_features_train = emoji_pca.fit_transform(emoji_tfidf_features)
    print("Emoji Explained Variance: ", sum(
        emoji_pca.explained_variance_ratio_))
    profanity_features_train = profanity_pca.fit_transform(
        profanity_tfidf_features)
    print("Profanity Explained Variance: ", sum(
        profanity_pca.explained_variance_ratio_))
    word_features_train = word_pca.fit_transform(words_tfidf_features)
    print("Word Explained Variance: ", sum(word_pca.explained_variance_ratio_))

    return emoji_features_train, profanity_features_train, word_features_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf


def get_features_train(author_list_train, emoji_pca_dim=4, profanity_pca_dim=14, word_pca_dim=20, label="Generating Train Features", verbose=False):
    """
        Generate features for training. This means it will fit PCA/TFIDF by using get_features_train_pca
        Inputs:
            author_list_test - list of list of strings: Author List with Tweets to be transformed into an input vector
            emoji_pca_dim (int): Number of dimensions for the emoji pca: default=4
            profanity_pca_dim (int): Number of dimensions for the profanity pca: default=14
            word_pca_dim (int): Number of dimensions for the word pca: default=20
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            x_train : np.array N Author x All Features, to be used to train the calssifier 
            emoji_pca : Emoji PCA object (sklearn)
            profanity_pca : Profanity PCA object (sklearn)
            word_pca : Word PCA object (sklearn)
            emoji_tfidf : Emoji TFIDF object (tfidf.py)
            profanity_tfidf : Profanity TFIDF object (tfidf.py)
            words_tfidf : Word TFIDF object (tfidf.py)
    """
    print(label)
    pos_features, count_features, lix_features, sent_features, sep_punct_features = get_features_no_pca(
        author_list_train)
    emoji_features_train, profanity_features_train, word_features_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train_pca(
        author_list_train, emoji_pca_dim, profanity_pca_dim, word_pca_dim, label, verbose)

    x_train = np.concatenate((pos_features, count_features, sent_features, sep_punct_features,
                             lix_features, emoji_features_train, profanity_features_train, word_features_train), axis=1)

    return x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf


def generate_features_train_predict(train, train_labels, test, classifier_class=RandomForestClassifier(), emoji_pca_dim=4,
                                    profanity_pca_dim=14, word_pca_dim=20, label="", verbose=False):
    """
        Generate features for training and trains a classifier (sklearn) and tests it on a training set. 
        This means it will fit PCA/TFIDF by using get_features_train_pca and then uses the classifier class provided
        to fit a classifier and returns it.
        Inputs:
            train - list of list of strings: Author List with Tweets to use as train data
            train_labels - list the same size of train with either 'I' or 'NI' for each author
            test - list of list of strings: Author List with Tweets to use as test data
            classifier_class - SKlearn object to perform classification
            emoji_pca_dim (int): Number of dimensions for the emoji pca: default=4
            profanity_pca_dim (int): Number of dimensions for the profanity pca: default=14
            word_pca_dim (int): Number of dimensions for the word pca: default=20
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            classifier.predict(x_test): Predictions on the test data 
            classifier.predict(x_train): Predictions on the training data
            classifier : Sklearn Classifier object 
    """
    print(label)

    x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(
        train, emoji_pca_dim, profanity_pca_dim, word_pca_dim, label="", verbose=verbose)
    classifier = classifier_class
    classifier.fit(x_train, train_labels)

    x_test = get_features_test(
        test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)

    return classifier.predict(x_test), classifier.predict(x_train), classifier


def train_model(train, train_labels, classifier_class=RandomForestClassifier(), emoji_pca_dim=4,
                profanity_pca_dim=14, word_pca_dim=20, label="", verbose=False):
    """
        Generate features for training and trains a classifier (sklearn). 
        This means it will fit PCA/TFIDF by using get_features_train_pca and then uses the classifier class provided
        to fit a classifier and returns it.
        Inputs:
            train - list of list of strings: Author List with Tweets to use as train data
            train_labels - list the same size of train with either 'I' or 'NI' for each author
            classifier_class - SKlearn object to perform classification
            emoji_pca_dim (int): Number of dimensions for the emoji pca: default=4
            profanity_pca_dim (int): Number of dimensions for the profanity pca: default=14
            word_pca_dim (int): Number of dimensions for the word pca: default=20
            label(str): Used when printing the final to show the output
            verbose(bool): Flag to control the log level
        Output:
            classifier: Sklearn object fitted to train
            emoji_pca : Emoji PCA object (sklearn)
            profanity_pca : Profanity PCA object (sklearn)
            word_pca : Word PCA object (sklearn)
            emoji_tfidf : Emoji TFIDF object (tfidf.py)
            profanity_tfidf : Profanity TFIDF object (tfidf.py)
            words_tfidf : Word TFIDF object (tfidf.py)
            x_train : np.array N Author x All Features, to be used to train the calssifier 
    """
    print(label)

    pos_features = get_features(
        train, pos_counts, "Individual Predict", verbose=verbose)
    count_features = get_features(
        train, author_style_counts, "Individual Predict", verbose=verbose)
    lix_features = get_features(
        train, lix_score, "Individual Predict", verbose=verbose)
    sent_features = get_features(
        train, get_sent_polarity, "Individual Predict", verbose=verbose)
    sep_punct_features = get_features(
        train, seperated_punctuation, "Individual Predict", verbose=verbose)

    emoji_pca_n = emoji_pca_dim
    profanity_pca_n = profanity_pca_dim
    word_pca_n = word_pca_dim

    emoji_pca = PCA(n_components=emoji_pca_n)
    profanity_pca = PCA(n_components=profanity_pca_n)
    word_pca = PCA(n_components=word_pca_n)

    emoji_tfidf = fit_emoji_embeds_tfidf(train)
    emoji_tfidf_features = get_features(
        train, emoji_tfidf.tf_idf, "Emoji TF_IDF")

    profanity_tfidf = fit_profanity_embeds_tfidf(train)
    profanity_tfidf_features = get_features(
        train, profanity_tfidf.tf_idf, "Profanity TF_IDF")

    words_tfidf = fit_word_embeds_tfidf(train)
    words_tfidf_features = get_features(
        train, words_tfidf.tf_idf, "Words TF_IDF")

    emoji_features_train = emoji_pca.fit_transform(emoji_tfidf_features)
    profanity_features_train = profanity_pca.fit_transform(
        profanity_tfidf_features)
    word_features_train = word_pca.fit_transform(words_tfidf_features)

    x_train = np.concatenate((pos_features, count_features, sent_features, sep_punct_features,
                             lix_features, emoji_features_train, profanity_features_train, word_features_train), axis=1)

    classifier = classifier_class
    classifier.fit(x_train, train_labels)

    return classifier, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, x_train


def predict(test, classifier, emoji_pca, profanity_pca, word_pca,
            emoji_tfidf, profanity_tfidf, words_tfidf):
    """
        Generate test features and predicts using the classifier (needs to be trained). 
        Inputs:
            test - list of list of strings: Author List with Tweets to be predicted
            classifier - SKlearn object to perform classification
            emoji_pca, profanity_pca, word_pca: pca objects trained on the training data, 
            these can be obtained when running get_features_train
            emoji_tfidf, profanity_tfidf, words_tfidf: tfidf objects trained on the trainind data, 
            these can be obtained when running get_features_train
        Output:
            classifier.predict(x_test) : list of predictions ('I' or 'NI')
            classifier.predict_proba(x_test) : list of predictions probabilities
    """
    x_test = get_features_test(test, emoji_pca, profanity_pca, word_pca,
                               emoji_tfidf, profanity_tfidf, words_tfidf, verbose=False)

    return classifier.predict(x_test), classifier.predict_proba(x_test)


def cache_features(X, REGEN_FEATURES=False):
    """
        Creates a file on disk with the transformed features. Can be useful if the same
        data is being used again and again.  
        Inputs:
            X - list of authors to create features for
            REGEN_FEATURES - Force flag to ensure that if the file is found it's overwritten.
        Output:
            VOID METHOD (no output)
            Creates the file in current directory with the name based on class for features.
    """
    if not REGEN_FEATURES and os.path.isfile("pos_features.csv"):
        pass
    else:
        pos_features = get_features(X, pos_counts, "All Data")
        np.savetxt("pos_features.csv", pos_features, delimiter=",", fmt='%f')

    # author style
    if not REGEN_FEATURES and os.path.isfile("author_style_counts.csv"):
        pass
    else:
        count_features = get_features(X, author_style_counts, "All Data")
        np.savetxt("author_style_counts.csv",
                   count_features, delimiter=",", fmt='%f')

    # lix
    if not REGEN_FEATURES and os.path.isfile("lix_score.csv"):
        pass
    else:
        lix_features = get_features(X, lix_score, "All Data")
        np.savetxt("lix_score.csv", lix_features, delimiter=",", fmt='%f')

    # punctuation
    if not REGEN_FEATURES and os.path.isfile("punct_score.csv"):
        pass
    else:
        punct_features = get_features(X, seperated_punctuation, "All Data")
        np.savetxt("sep_punct_score.csv", punct_features,
                   delimiter=",", fmt='%f')

    # seperated pronunciation
    if not REGEN_FEATURES and os.path.isfile("sep_punct_score.csv"):
        pass
    else:
        sep_punct_features = get_features(X, seperated_punctuation, "All Data")
        np.savetxt("sep_punct_score.csv", sep_punct_features,
                   delimiter=",", fmt='%f')

    # emoji features
    if not REGEN_FEATURES and os.path.isfile("emoji_features.csv"):
        pass
    else:
        emoji_features = get_features(X, emoji_embeds, "All Data")
        np.savetxt("emoji_features.csv",  emoji_features,
                   delimiter=",", fmt='%f')

    # sentence polarity
    if not REGEN_FEATURES and os.path.isfile("get_sent_polarity.csv"):
        pass
    else:
        sent_features = get_features(X, get_sent_polarity, "All Data")
        np.savetxt("get_sent_polarity.csv",
                   sent_features, delimiter=",", fmt='%f')

    # profanity
    if not REGEN_FEATURES and os.path.isfile("profanity_counts.csv"):
        pass
    else:
        profanity_features = get_features(X, profanity_embeds, "All Data")
        np.savetxt("profanity_counts.csv",  profanity_features,
                   delimiter=",", fmt='%f')


def cross_validate(X, y, split_n=7, emoji_pca_dim=4, profanity_pca_dim=14, word_pca_dim=20,
                   rdmforest=True, one_nn=True, three_nn=True,
                   five_nn=True, log_cls=True, svm_cls=True):
    """
        Performs cross-validation using specified splits.
        The method ensures that no training data is used to fit the PCA and TF-IDF. However, this means that
        for each split the TF-IDF/PCA features needs to be re-calculated, which takes time.

        Inputs:
            X - list of authors with tweets to be used as features
            y - list the same size of X with either 'I' or 'NI' for each author
            split_n (int) - number of cross validation splits to perform (default = 7)
            emoji_pca_dim (int): Number of dimensions for the emoji pca: default=4
            profanity_pca_dim (int): Number of dimensions for the profanity pca: default=14
            word_pca_dim (int): Number of dimensions for the word pca: default=20
            rdmforest (bool), ... : Defines if the classifier should be used during cross validation
        Output:
            acc_dict (dict): of type "Classifier" -> [(TrainSplit1_acc, TestSplit1_Acc), (TrainSplit2_acc, TestSplit2_Acc)...]
            f1_dict: of type "Classifier" -> [F1ScoreTestSplit1, F1ScoreTestSplit2, ...]
    """
    if rdmforest:
        random_forest_list = []
        f1_random_forest_list = []
    if one_nn:
        one_nn_list = []
        f1_one_nn_list = []
    if three_nn:
        three_nn_list = []
        f1_three_nn_list = []
    if five_nn:
        five_nn_list = []
        f1_five_nn_list = []
    if log_cls:
        log_reg_list = []
        f1_log_reg_list = []
    if svm_cls:
        svm_list = []
        f1_svm_list = []

    importance_mean  = dict()
    importance_std = dict()
    stop_w = stopwatch()
    kf = KFold(n_splits=split_n)
    print(f"Performing Cross-Validation...")
    # Save time by not re-computing static features.
    pos_features, count_features, lix_features, sent_features, sep_punct_features = get_features_no_pca(
        X)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        ratio_train_i = (y_train == "I").sum()/len(y_train)
        ratio_train_ni = (y_train == "NI").sum()/len(y_train)

        ratio_test_i = (y_test == "I").sum()/len(y_test)
        ratio_test_ni = (y_test == "NI").sum()/len(y_test)

        print(f"(split: {i+1}/{split_n}) Train: ",
              len(train_index), "Test: ", len(test_index))
        print(train_index[:10])
        print(test_index[:10])
        if i > 0:
            print(
                f"{stop_w.current_time()}: Split took {stop_w.split(as_string=True)}")
        print(
            f"Train Ratio of Labels (split: {i+1}/{split_n}): I: {ratio_train_i} | NI:{ratio_train_ni}")
        print(
            f"Test Ratio of Labels (split: {i+1}/{split_n}) : I: {ratio_test_i} | NI:{ratio_test_ni}")

        emoji_features_train, profanity_features_train, word_features_train, *trained_features = get_features_train_pca(X_train, label="Generating Train Features",
                                                                                                                            emoji_pca_dim=emoji_pca_dim, profanity_pca_dim=profanity_pca_dim, word_pca_dim=word_pca_dim)
        emoji_pca, profanity_pca, word_pca, *tf_idf_features = trained_features
        emoji_tfidf, profanity_tfidf, words_tfidf = tf_idf_features

        X_train = np.concatenate((pos_features[train_index], count_features[train_index], sent_features[train_index], sep_punct_features[train_index],
                                  lix_features[train_index], emoji_features_train, profanity_features_train, word_features_train), axis=1)

        emoji_features_test, profanity_features_test, word_features_test = get_features_test_pca(X_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)

        X_test = np.concatenate((pos_features[test_index], count_features[test_index], sent_features[test_index], sep_punct_features[test_index],
                                  lix_features[test_index], emoji_features_test, profanity_features_test, word_features_test), axis=1)

        #print("Word features shape: ", word_features_train.shape)

        if rdmforest:
            clf_rfc = RandomForestClassifier()
            clf_rfc.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_rfc.predict(
                X_test), clf_rfc.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)

            f1_random_forest_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            random_forest_list.append((acc_train, acc_test))

            print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
            print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")
            #feature importance
            perm_tree = permutation_importance(clf_rfc, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["RandomForest"] = importance_mean.get("RandomForest", 0) + perm_tree.importances_mean
            importance_std["RandomForest"] = importance_std.get("RandomForest", 0) + perm_tree.importances_std
        if one_nn:
            clf_nn1 =make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
            clf_nn1.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn1.predict(
                X_test), clf_nn1.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            one_nn_list.append((acc_train, acc_test))
            f1_one_nn_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            print(f"1-NN acc (Train): {acc_train:.4f}")
            print(f"1-NN (Test)     : {acc_test:.4f}")

            #feature importance
            perm_nn1 = permutation_importance(clf_nn1, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["1-NN"] = importance_mean.get("1-NN", 0) + perm_nn1.importances_mean
            importance_std["1-NN"] = importance_std.get("1-NN", 0) + perm_nn1.importances_std

        if three_nn:
            clf_nn3 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
            clf_nn3.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn3.predict(
                X_test), clf_nn3.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            three_nn_list.append((acc_train, acc_test))
            f1_three_nn_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            print(f"3-NN acc (Train): {acc_train:.4f}")
            print(f"3-NN (Test)     : {acc_test:.4f}")
            #feature importance
            perm_nn3 = permutation_importance(clf_nn3, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["3-NN"] = importance_mean.get("3-NN", 0) + perm_nn3.importances_mean
            importance_std["3-NN"] = importance_std.get("3-NN", 0) + perm_nn3.importances_std

        if five_nn:
            clf_nn5 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
            clf_nn5.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn5.predict(
                X_test), clf_nn5.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            five_nn_list.append((acc_train, acc_test))
            f1_five_nn_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            print(f"5-NN acc (Train): {acc_train:.4f}")
            print(f"5-NN (Test)     : {acc_test:.4f}")

            #feature importance
            perm_nn5 = permutation_importance(clf_nn5, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["5-NN"] = importance_mean.get("5-NN", 0) + perm_nn5.importances_mean
            importance_std["5-NN"] = importance_std.get("5-NN", 0) + perm_nn5.importances_std

        if svm_cls:
            pipe_svm = make_pipeline(StandardScaler(), svm.SVC(gamma="auto"))
            pipe_svm.fit(X_train, y_train)
            y_test_pred, y_train_pred = pipe_svm.predict(
                X_test), pipe_svm.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)

            svm_list.append((acc_train, acc_test))
            f1_svm_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            print(f"SVM (Train): {acc_train:.4f}")
            print(f"SVM (Test) : {acc_test:.4f}")

            #feature importance
            perm_svm = permutation_importance(pipe_svm, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["SVM"] = importance_mean.get("SVM", 0) + perm_svm.importances_mean
            importance_std["SVM"] = importance_std.get("SVM", 0) + perm_svm.importances_std

        if log_cls:
            pipe_log = make_pipeline(StandardScaler(), LogisticRegression())
            pipe_log.fit(X_train, y_train)
            y_test_pred, y_train_pred = pipe_log.predict(
                X_test), pipe_log.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            log_reg_list.append((acc_train, acc_test))
            f1_log_reg_list.append(
                f1_score(y_test, y_test_pred, average='weighted'))
            print(f"Log-Reg (Train): {acc_train:.4f}")
            print(f"Log-Reg (Test) : {acc_test:.4f}")

            #feature importance
            perm_log = permutation_importance(pipe_log, X_test, y_test, n_repeats=5, n_jobs=5)
            importance_mean["LogRegression"] = importance_mean.get("LogRegression", 0) + perm_log.importances_mean
            importance_std["LogRegression"] = importance_std.get("LogRegression", 0) + perm_log.importances_std

    acc_dict = {}
    f1_dict = {}
    
    feature_names = [ 
    "ADJ","ADP","ADV","CONJ","DET",
    "NUM","PRT","PRON","VERB",
    "auth_vocabsize","type_token_rt","avg_author_word_length","avg_tweet_length","avg_author_hashtag_count",
    "avg_author_usertag_count","avg_author_urltag_count","author_avg_emoji","avg_capital_lower_ratio",
    "pos", "neut", "neg", "compound",
    "mult-ex", "mult-qu", "mult-pe", "quote", "ex", "qu", "pe",
    "LiXScore",
    "emoji_pca_1", "emoji_pca_2", "emoji_pca_3", "emoji_pca_4",
    "profanity_pca_1", "profanity_pca_2", "profanity_pca_3", "profanity_pca_4", "profanity_pca_5", 
    "profanity_pca_6","profanity_pca_7", "profanity_pca_8","profanity_pca_9","profanity_pca_10",
    "profanity_pca_11","profanity_pca_12","profanity_pca_13","profanity_pca_14",
    "word_pca1","word_pca2","word_pca3", "word_pca4", "word_pca5", "word_pca6", "word_pca7", "word_pca8",
    "word_pca9", "word_pca10", "word_pca11", "word_pca12", "word_pca13", "word_pca14","word_pca15","word_pca16",
    "word_pca17", "word_pca18" , "word_pca19", "word_pca20"
    ]
    
    #feature_names = np.arange(0, X_train.shape[1])

    

    if rdmforest:
        random_forest_list = np.array(random_forest_list)
        f1_random_forest_list = np.array(f1_random_forest_list)
        acc_dict['RandomForest'] = random_forest_list
        f1_dict['RandomForest'] = f1_random_forest_list

        importances = pd.Series(importance_mean["RandomForest"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["RandomForest"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (random Forests)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

    if one_nn:
        one_nn_list = np.array(one_nn_list)
        f1_one_nn_list = np.array(f1_one_nn_list)
        acc_dict['1-NN'] = one_nn_list
        f1_dict['1-NN'] = f1_one_nn_list

        importances = pd.Series(importance_mean["1-NN"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["1-NN"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (1-NN)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

    if three_nn:
        three_nn_list = np.array(three_nn_list)
        f1_three_nn_list = np.array(f1_three_nn_list)
        acc_dict['3-NN'] = three_nn_list
        f1_dict['3-NN'] = f1_three_nn_list

        importances = pd.Series(importance_mean["3-NN"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["3-NN"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (3-NN)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

    if five_nn:
        five_nn_list = np.array(five_nn_list)
        f1_five_nn_list = np.array(f1_five_nn_list)
        acc_dict['5-NN'] = five_nn_list
        f1_dict['5-NN'] = f1_five_nn_list

        importances = pd.Series(importance_mean["5-NN"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["5-NN"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (5-NN)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

    if log_cls:
        log_reg_list = np.array(log_reg_list)
        f1_log_reg_list = np.array(f1_log_reg_list)
        acc_dict['LogRegression'] = log_reg_list
        f1_dict['LogRegression'] = f1_log_reg_list

        importances = pd.Series(importance_mean["LogRegression"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["LogRegression"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (Log-Reg)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

    if svm_cls:
        svm_list = np.array(svm_list)
        f1_svm_list = np.array(f1_svm_list)
        acc_dict['SVM'] = svm_list
        f1_dict['SVM'] = f1_svm_list

        importances = pd.Series(importance_mean["SVM"]/split_n, index=feature_names)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr=importance_std["SVM"]/split_n, ax=ax)
        ax.set_title("Feature importances using permutation on full model (SVM)")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
        

    return acc_dict, f1_dict

def cross_validate_tune_params(X, y, split_n=7, emoji_pca_dim=[5], profanity_pca_dim=[10], word_pca_dim=[20],
                   rdmforest=True, one_nn=True, three_nn=True,
                   five_nn=True, log_cls=True, svm_cls=True):
    """
        Same as cross_validate, but takes a list of dimensions for PCA to find the best parameters.

        Inputs:
            X - list of authors with tweets to be used as features
            y - list the same size of X with either 'I' or 'NI' for each author
            split_n (int) - number of cross validation splits to perform (default = 7)
            emoji_pca_dim (list(int)): List of number of dimensions for the emoji pca: default=4
            profanity_pca_dim (list(int)): List of number of dimensions for the profanity pca: default=14
            word_pca_dim (list(int)): List of number of dimensions for the word pca: default=20
            rdmforest (bool), ... : Defines if the classifier should be used during cross validation
        Output:
            acc_dict (dict): of type "Classifier" -> (emoji_dim, profanity_dim, word_dim) -> [(TrainSplit1_acc, TestSplit1_Acc), (TrainSplit2_acc, TestSplit2_Acc)...]
            f1_dict: of type "Classifier" -> (emoji_dim, profanity_dim, word_dim) -> [F1ScoreTestSplit1, F1ScoreTestSplit2, ...]
            best_e, best_p, best_w : Best parameters for emoji, profanity, word parameters on acc 
    """
    acc_dict = {}
    f1_dict = {}

    if rdmforest:
        acc_dict['RandomForest'] = dict()
        f1_dict['RandomForest']  = dict()
    if one_nn:
        acc_dict['1-NN'] = dict()
        f1_dict['1-NN'] = dict()
    if three_nn:
        acc_dict['3-NN'] = dict()
        f1_dict['3-NN'] = dict()
    if five_nn:
        acc_dict['5-NN'] = dict()
        f1_dict['5-NN'] = dict()
    if log_cls:
        acc_dict['LogRegression'] = dict()
        f1_dict['LogRegression'] = dict()
    if svm_cls:
        acc_dict['SVM'] = dict()
        f1_dict['SVM'] = dict()

    stop_w = stopwatch()
    kf = KFold(n_splits=split_n)
    print(f"Performing Cross-Validation...")
    # Save time by not re-computing static features.
    pos_features, count_features, lix_features, sent_features, sep_punct_features = get_features_no_pca(
        X)
    best_e, best_p, best_w = -1,-1,-1
    for emoji_n in emoji_pca_dim:
        for profanity_n in profanity_pca_dim:
            for word_n in word_pca_dim:
                params_key = (emoji_n, profanity_n, word_n)
                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    
                    X_train, X_test = X[train_index], X[test_index]

                    y_train, y_test = y[train_index], y[test_index]

                    ratio_train_i = (y_train == "I").sum()/len(y_train)
                    ratio_train_ni = (y_train == "NI").sum()/len(y_train)

                    ratio_test_i = (y_test == "I").sum()/len(y_test)
                    ratio_test_ni = (y_test == "NI").sum()/len(y_test)

                    print(f"(split: {i+1}/{split_n}) Train: ",
                        len(train_index), "Test: ", len(test_index), " | Params: ", (emoji_n, profanity_n, word_n))
                    print(f"Best params so far: {best_e, best_p, best_w}")
                    print(train_index[:10])
                    print(test_index[:10])
                    if i > 0:
                        print(
                            f"{stop_w.current_time()}: Split took {stop_w.split(as_string=True)}, total: {stop_w.elapsed(True)}")
                    print(
                        f"Train Ratio of Labels (split: {i+1}/{split_n}): I: {ratio_train_i} | NI:{ratio_train_ni}")
                    print(
                        f"Test Ratio of Labels (split: {i+1}/{split_n}) : I: {ratio_test_i} | NI:{ratio_test_ni}")

                    emoji_features_train, profanity_features_train, word_features_train, *trained_features = get_features_train_pca(X_train, label="Generating Train Features",
                                                                                                                                        emoji_pca_dim=emoji_n, profanity_pca_dim=profanity_n, word_pca_dim=word_n)
                    emoji_pca, profanity_pca, word_pca, *tf_idf_features = trained_features
                    emoji_tfidf, profanity_tfidf, words_tfidf = tf_idf_features

                    X_train = np.concatenate((pos_features[train_index], count_features[train_index], sent_features[train_index], sep_punct_features[train_index],
                                            lix_features[train_index], emoji_features_train, profanity_features_train, word_features_train), axis=1)

                    emoji_features_test, profanity_features_test, word_features_test = get_features_test_pca(X_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)

                    X_test = np.concatenate((pos_features[test_index], count_features[test_index], sent_features[test_index], sep_punct_features[test_index],
                                            lix_features[test_index], emoji_features_test, profanity_features_test, word_features_test), axis=1)

                    
                    if rdmforest:
                        clf_rfc = RandomForestClassifier()
                        clf_rfc.fit(X_train, y_train)
                        y_test_pred, y_train_pred = clf_rfc.predict(
                            X_test), clf_rfc.predict(X_train)

                        acc_train = np.sum(y_train_pred == y_train) / len(y_train)
                        acc_test = np.sum(y_test_pred == y_test) / len(y_test)

                        acc_dict['RandomForest'][params_key] = acc_dict['RandomForest'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['RandomForest'][params_key] = f1_dict['RandomForest'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]

                        print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
                        print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")

                    if one_nn:
                        clf_nn1 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
                        clf_nn1.fit(X_train, y_train)
                        y_test_pred, y_train_pred = clf_nn1.predict(
                            X_test), clf_nn1.predict(X_train)

                        acc_train = np.sum(y_train_pred == y_train) / len(y_train)
                        acc_test = np.sum(y_test_pred == y_test) / len(y_test)

                        acc_dict['1-NN'][params_key] = acc_dict['1-NN'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['1-NN'][params_key] =  f1_dict['1-NN'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]

                        print(f"1-NN acc (Train): {acc_train:.4f}")
                        print(f"1-NN (Test)     : {acc_test:.4f}")

                    if three_nn:
                        clf_nn3 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
                        clf_nn3.fit(X_train, y_train)
                        y_test_pred, y_train_pred = clf_nn3.predict(
                            X_test), clf_nn3.predict(X_train)

                        acc_train = np.sum(y_train_pred == y_train) / len(y_train)
                        acc_test = np.sum(y_test_pred == y_test) / len(y_test)

                        acc_dict['3-NN'][params_key] = acc_dict['3-NN'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['3-NN'][params_key] = f1_dict['3-NN'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]

                        print(f"3-NN acc (Train): {acc_train:.4f}")
                        print(f"3-NN (Test)     : {acc_test:.4f}")

                    if five_nn:
                        clf_nn5 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
                        clf_nn5.fit(X_train, y_train)
                        y_test_pred, y_train_pred = clf_nn5.predict(
                            X_test), clf_nn5.predict(X_train)

                        acc_train = np.sum(y_train_pred == y_train) / len(y_train)
                        acc_test = np.sum(y_test_pred == y_test) / len(y_test)

                        acc_dict['5-NN'][params_key] = acc_dict['5-NN'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['5-NN'][params_key] = f1_dict['5-NN'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]

                        print(f"5-NN acc (Train): {acc_train:.4f}")
                        print(f"5-NN (Test)     : {acc_test:.4f}")

                    if svm_cls:
                        pipe_svm = make_pipeline(StandardScaler(), svm.SVC(gamma="auto"))
                        pipe_svm.fit(X_train, y_train)
                        y_test_pred, y_train_pred = pipe_svm.predict(
                            X_test), pipe_svm.predict(X_train)

                        acc_train = np.sum(y_train_pred == y_train) / len(y_train)
                        acc_test = np.sum(y_test_pred == y_test) / len(y_test)

                        acc_dict['SVM'][params_key] = acc_dict['SVM'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['SVM'][params_key] = f1_dict['SVM'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]
                        print(f"SVM (Train): {acc_train:.4f}")
                        print(f"SVM (Test) : {acc_test:.4f}")

                    if log_cls:
                        pipe_log = make_pipeline(StandardScaler(), LogisticRegression())
                        pipe_log.fit(X_train, y_train)
                        y_test_pred, y_train_pred = pipe_log.predict(
                            X_test), pipe_log.predict(X_train)

                        acc_dict['LogRegression'][params_key] = acc_dict['LogRegression'].get(params_key, []) + [[acc_train, acc_test]]
                        f1_dict['LogRegression'][params_key] = f1_dict['LogRegression'].get(params_key, []) + [[f1_score(y_test, y_test_pred, average='weighted')]]
                        print(f"Log-Reg (Train): {acc_train:.4f}")
                        print(f"Log-Reg (Test) : {acc_test:.4f}")

                currrent_max = -1
                current_best_params = -1
                for class_keys in acc_dict.keys():
                    for keys in acc_dict[class_keys].keys():
                        acc_dict[class_keys][keys] = np.array(acc_dict[class_keys][keys])
                        f1_dict[class_keys][keys] = np.array(f1_dict[class_keys][keys])
                        avg_vals = acc_dict[class_keys][keys].mean(axis=0)
                        if currrent_max < avg_vals[1]:
                            currrent_max = avg_vals[1]
                            current_best_params = keys
                best_e, best_p, best_w = current_best_params
                print("------------------------------------------------------------------------------------------------------")
                print(f"Best Params (ACC) so far - {currrent_max} : Emoji_n:{best_e} | Profanity_n:{best_p} | Word_n:{best_w}")
                print("------------------------------------------------------------------------------------------------------")
    return acc_dict, f1_dict, best_e, best_p, best_w

def print_dictionaries_cross_validate(dict_acc, dict_f1):
    """
        Simple method to print the mean results for each dictionary resulting from cross_validation.

        Note: this does not work for the tuning_cross_validation.
    """
    keys = list(dict_acc.keys())
    n_splits = len(dict_acc[keys[0]])
    print(f"Printing values for n=={n_splits}")
    for key in dict_acc.keys():
        print(
              f"Average acc,           for {key}: {dict_acc[key].mean(axis=0)}")
    for key in dict_acc.keys():
        print(f"Average F1 (Weighted), for {key}: {dict_f1[key].mean(axis=0)}")
