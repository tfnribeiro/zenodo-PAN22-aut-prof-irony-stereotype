from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, SparsePCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn import svm
from pos_counts import *
from count_features import *
from lexical_comp import *
from sent_polarity import *
from read_files import *
from punctuation import *
from tqdm import tqdm
import pandas as pd
import os 

np.random.seed(0)
def get_features(dataset, function, label="", supress_print=False):
    list_features = []
    if supress_print:
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
def get_features_test(author_list_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, supress_prints_flag=False):
    
    pos_features = get_features(author_list_test, pos_counts, "Individual Predict", supress_print=supress_prints_flag)
    count_features = get_features(author_list_test, author_style_counts, "Individual Predict", supress_print=supress_prints_flag)
    lix_features = get_features(author_list_test, lix_score, "Individual Predict", supress_print=supress_prints_flag)
    sent_features = get_features(author_list_test, get_sent_polarity, "Individual Predict", supress_print=supress_prints_flag)
    sep_punct_features = get_features(author_list_test, count_punctuation, "Individual Predict", supress_print=supress_prints_flag)
    #miss_features = get_features(test, misspelled, "Individual Predict", supress_print=True).reshape((-1,1))
    emoji_tfidf_features = get_features(author_list_test, emoji_tfidf.tf_idf, "Individual Predict", supress_print=supress_prints_flag)
    profanity_tfidf_features = get_features(author_list_test, profanity_tfidf.tf_idf, "Individual Predict", supress_print=supress_prints_flag)
    words_tfidf_features = get_features(author_list_test, words_tfidf.tf_idf, "Words TF_IDF", supress_print=supress_prints_flag)
    emoji_features_test = emoji_pca.transform(emoji_tfidf_features)
    profanity_features_test = profanity_pca.transform(profanity_tfidf_features)
    word_features_test = word_pca.transform(words_tfidf_features)

    x_test = np.concatenate((pos_features, count_features, sent_features, sep_punct_features, lix_features, emoji_features_test, profanity_features_test, word_features_test), axis=1)

    return x_test

def get_features_train(author_list_train, emoji_pca_dim = 5, profanity_pca_dim = 10, word_pca_dim = 20, label="", supress_prints_flag=False):
    pos_features = get_features(author_list_train, pos_counts, "Individual Predict", supress_print=supress_prints_flag)
    count_features = get_features(author_list_train, author_style_counts, "Individual Predict", supress_print=supress_prints_flag)
    lix_features = get_features(author_list_train, lix_score, "Individual Predict", supress_print=supress_prints_flag)
    sent_features = get_features(author_list_train, get_sent_polarity, "Individual Predict", supress_print=supress_prints_flag)
    sep_punct_features = get_features(author_list_train, count_punctuation, "Individual Predict", supress_print=supress_prints_flag)
    #miss_features = get_features(train, misspelled, "Individual Predict", supress_print=True).reshape((-1,1))
    #emoji_features = get_features(train, emoji_embeds, "Individual Predict", supress_print=True)
    #profanity_features = get_features(train, profanity_embeds, "Individual Predict", supress_print=True)
    emoji_pca_n = emoji_pca_dim 
    profanity_pca_n = profanity_pca_dim
    word_pca_n = word_pca_dim

    emoji_pca = PCA(n_components=emoji_pca_n)
    profanity_pca = PCA(n_components=profanity_pca_n)
    word_pca = PCA(n_components=word_pca_n)

    emoji_tfidf = fit_emoji_embeds_tfidf(author_list_train)
    emoji_tfidf_features = get_features(author_list_train, emoji_tfidf.tf_idf, "Emoji TF_IDF")

    profanity_tfidf = fit_profanity_embeds_tfidf(author_list_train)
    profanity_tfidf_features = get_features(author_list_train, profanity_tfidf.tf_idf, "Profanity TF_IDF")

    words_tfidf = fit_word_embeds_tfidf(author_list_train)
    words_tfidf_features = get_features(author_list_train, words_tfidf.tf_idf, "Words TF_IDF")

    emoji_features_train = emoji_pca.fit_transform(emoji_tfidf_features)
    profanity_features_train = profanity_pca.fit_transform(profanity_tfidf_features)
    word_features_train = word_pca.fit_transform(words_tfidf_features)

    x_train = np.concatenate((pos_features, count_features, sent_features, sep_punct_features, lix_features, emoji_features_train, profanity_features_train, word_features_train), axis=1)

    return x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf

def generate_features_train_predict(train, train_labels, test, classifier_class=RandomForestClassifier(), emoji_pca_dim = 5, 
    profanity_pca_dim = 10, word_pca_dim = 20, label="", supress_prints_flag=False):

    print(label)

    x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(train, emoji_pca_dim, profanity_pca_dim, word_pca_dim, label="", supress_prints_flag=False)
    classifier = classifier_class
    classifier.fit(x_train, train_labels)

    x_test = get_features_test(test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)

    return classifier.predict(x_test), classifier.predict(x_train), classifier.predict_proba(x_test)

def train_model(train, train_labels, classifier_class=RandomForestClassifier(), emoji_pca_dim = 5, 
    profanity_pca_dim = 10, word_pca_dim = 20, label="", supress_prints_flag=False):

    print(label)
    
    pos_features = get_features(train, pos_counts, "Individual Predict", supress_print=supress_prints_flag)
    count_features = get_features(train, author_style_counts, "Individual Predict", supress_print=supress_prints_flag)
    lix_features = get_features(train, lix_score, "Individual Predict", supress_print=supress_prints_flag)
    sent_features = get_features(train, get_sent_polarity, "Individual Predict", supress_print=supress_prints_flag)
    sep_punct_features = get_features(train, count_punctuation, "Individual Predict", supress_print=supress_prints_flag)

    emoji_pca_n = emoji_pca_dim 
    profanity_pca_n = profanity_pca_dim
    word_pca_n = word_pca_dim

    emoji_pca = PCA(n_components=emoji_pca_n)
    profanity_pca = PCA(n_components=profanity_pca_n)
    word_pca = PCA(n_components=word_pca_n)

    emoji_tfidf = fit_emoji_embeds_tfidf(train)
    emoji_tfidf_features = get_features(train, emoji_tfidf.tf_idf, "Emoji TF_IDF")

    profanity_tfidf = fit_profanity_embeds_tfidf(train)
    profanity_tfidf_features = get_features(train, profanity_tfidf.tf_idf, "Profanity TF_IDF")

    #emoji_features = get_features(X_train_all, emoji_embeds, "All Data")
    words_tfidf = fit_word_embeds_tfidf(train)
    words_tfidf_features = get_features(train, words_tfidf.tf_idf, "Words TF_IDF")

    emoji_features_train = emoji_pca.fit_transform(emoji_tfidf_features)
    profanity_features_train = profanity_pca.fit_transform(profanity_tfidf_features)
    word_features_train = word_pca.fit_transform(words_tfidf_features)


    x_train = np.concatenate((pos_features, count_features, sent_features, sep_punct_features, lix_features, emoji_features_train, profanity_features_train, word_features_train), axis=1)

    classifier = classifier_class
    classifier.fit(x_train, train_labels)
    
    return classifier, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf



def cache_features(REGEN_FEATURES=False):
    if not REGEN_FEATURES and os.path.isfile("pos_features.csv"):
        pos_features = np.loadtxt("pos_features.csv", delimiter=",")
    else:
        pos_features = get_features(X, pos_counts, "All Data")
        np.savetxt("pos_features.csv", pos_features, delimiter=",", fmt='%f')

    #author style
    if not REGEN_FEATURES and os.path.isfile("author_style_counts.csv"):
        print(get_author_style_labels())
        count_features = np.loadtxt("author_style_counts.csv", delimiter=",")
    else:
        count_features = get_features(X, author_style_counts, "All Data")
        np.savetxt("author_style_counts.csv", count_features, delimiter=",", fmt='%f')    

    #lix
    if not REGEN_FEATURES and os.path.isfile("lix_score.csv"):
        lix_features = np.loadtxt("lix_score.csv", delimiter=",").reshape((-1,1))
    else:
        lix_features = get_features(X, lix_score, "All Data")
        np.savetxt("lix_score.csv", lix_features, delimiter=",", fmt='%f')

    #punctuation
    if not REGEN_FEATURES and os.path.isfile("punct_score.csv"):
        punct_features = np.loadtxt("punct_score.csv", delimiter=",")
    else:
        punct_features = get_features(X, count_punctuation, "All Data")
        np.savetxt("punct_score.csv", punct_features, delimiter=",", fmt='%f')

    #seperated pronunciation
    if not REGEN_FEATURES and os.path.isfile("sep_punct_score.csv"):
        sep_punct_features = np.loadtxt("sep_punct_score.csv", delimiter=",")
    else:
        sep_punct_features = get_features(X, seperated_punctuation, "All Data")
        np.savetxt("sep_punct_score.csv", sep_punct_features, delimiter=",", fmt='%f')

    #emoji features
    if not REGEN_FEATURES and os.path.isfile("emoji_features.csv"):
        emoji_features = np.loadtxt("emoji_features.csv", delimiter=",")
    else:
        emoji_features = get_features(X, emoji_embeds, "All Data")
        np.savetxt("emoji_features.csv",  emoji_features, delimiter=",", fmt='%f')

    #sentence polarity
    if not REGEN_FEATURES and os.path.isfile("get_sent_polarity.csv"):
        sent_features = np.loadtxt("get_sent_polarity.csv", delimiter=",")
    else:
        sent_features = get_features(X, get_sent_polarity, "All Data")
        np.savetxt("get_sent_polarity.csv", sent_features, delimiter=",", fmt='%f')

    #profanity
    if not REGEN_FEATURES and os.path.isfile("profanity_counts.csv"):
        profanity_features = np.loadtxt("profanity_counts.csv", delimiter=",")    
    else:
        profanity_features = get_features(X, profanity_embeds, "All Data")
        np.savetxt("profanity_counts.csv",  profanity_features, delimiter=",", fmt='%f')

def predict(test, classifier, emoji_pca, profanity_pca, word_pca, 
    emoji_tfidf, profanity_tfidf, words_tfidf):

    pos_features = get_features(test, pos_counts, "Individual Predict", supress_print=supress_prints_flag)
    count_features = get_features(test, author_style_counts, "Individual Predict", supress_print=supress_prints_flag)
    lix_features = get_features(test, lix_score, "Individual Predict", supress_print=supress_prints_flag)
    sent_features = get_features(test, get_sent_polarity, "Individual Predict", supress_print=supress_prints_flag)
    sep_punct_features = get_features(test, count_punctuation, "Individual Predict", supress_print=supress_prints_flag)
    #miss_features = get_features(test, misspelled, "Individual Predict", supress_print=True).reshape((-1,1))
    emoji_tfidf_features = get_features(test, emoji_tfidf.tf_idf, "Individual Predict", supress_print=supress_prints_flag)
    profanity_tfidf_features = get_features(test, profanity_tfidf.tf_idf, "Individual Predict", supress_print=supress_prints_flag)
    words_tfidf_features = get_features(test, words_tfidf.tf_idf, "Words TF_IDF", supress_print=supress_prints_flag)
    emoji_features_test = emoji_pca.transform(emoji_tfidf_features)
    profanity_features_test = profanity_pca.transform(profanity_tfidf_features)
    word_features_test = word_pca.transform(words_tfidf_features)

    x_test = np.concatenate((pos_features, count_features, sent_features, sep_punct_features, lix_features, emoji_features_test, profanity_features_test, word_features_test), axis=1)

    return classifier.predict(x_test), classifier.predict_proba(x_test)

def cross_validate(split_n = 7, rdmforest=True, one_nn=True, three_nn=True,
    five_nn=True, log_cls=True, svm_cls=True):
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

    kf = KFold(n_splits=split_n)
    print(f"Performing Cross-Validation...")
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Train: ", len(train_index), "Test: ", len(test_index))
        
        X_train, X_test = X[train_index], X[test_index]
        
        y_train, y_test = y[train_index], y[test_index]

        X_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X_train,label="Generating Train Features")
        X_test = get_features_test(X_test, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)
        ratio_train_i = (y_train == "I").sum()/len(y_train)
        ratio_train_ni = (y_train == "NI").sum()/len(y_train)

        ratio_test_i = (y_test == "I").sum()/len(y_test)
        ratio_test_ni = (y_test == "NI").sum()/len(y_test)

        print(f"Train Ratio of Labels (split: {i+1}): I: {ratio_train_i} | NI:{ratio_train_ni}")
        print(f"Test Ratio of Labels (split: {i+1}): I: {ratio_test_i} | NI:{ratio_test_ni}")

        if rdmforest:
            clf_rfc = RandomForestClassifier()
            clf_rfc.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_rfc.predict(X_test), clf_rfc.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)

            f1_random_forest_list.append(f1_score(y_test, y_test_pred,average='weighted'))
            random_forest_list.append((acc_train,acc_test))

            print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
            print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")
        
        if one_nn:
            clf_nn1 = KNeighborsClassifier(n_neighbors=1)
            clf_nn1.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn1.predict(X_test), clf_nn1.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            one_nn_list.append((acc_train,acc_test))
            f1_one_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
            print(f"1-NN acc (Train): {acc_train:.4f}")
            print(f"1-NN (Test): {acc_test:.4f}")
        
        if three_nn:
            clf_nn3 = KNeighborsClassifier(n_neighbors=3)
            clf_nn3.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn3.predict(X_test), clf_nn3.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            three_nn_list.append((acc_train,acc_test))
            f1_three_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
            print(f"3-NN acc (Train): {acc_train:.4f}")
            print(f"3-NN (Test): {acc_test:.4f}")

        if five_nn:
            clf_nn5 = KNeighborsClassifier(n_neighbors=5)
            clf_nn5.fit(X_train, y_train)
            y_test_pred, y_train_pred = clf_nn5.predict(X_test), clf_nn5.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            five_nn_list.append((acc_train,acc_test))
            f1_five_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
            print(f"5-NN acc (Train): {acc_train:.4f}")
            print(f"5-NN (Test): {acc_test:.4f}")
        
        if svm_cls:
            pipe_svm = make_pipeline(StandardScaler(), svm.SVC(gamma="auto"))
            pipe_svm.fit(X_train, y_train)
            y_test_pred, y_train_pred = pipe_svm.predict(X_test), pipe_svm.predict(X_train)

            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)

            svm_list.append((acc_train,acc_test))
            f1_svm_list.append(f1_score(y_test, y_test_pred,average='weighted'))
        
        if log_cls:
            pipe_log = make_pipeline(StandardScaler(), LogisticRegression())
            pipe_log.fit(X_train, y_train)
            y_test_pred, y_train_pred = pipe_log.predict(X_test), pipe_log.predict(X_train)
            
            acc_train = np.sum(y_train_pred == y_train) / len(y_train)
            acc_test = np.sum(y_test_pred == y_test) / len(y_test)
            log_reg_list.append((acc_train,acc_test))
            f1_log_reg_list.append(f1_score(y_test, y_test_pred,average='weighted'))
        
        acc_dict = {}
        f1_dict = {}

        if rdmforest:
            random_forest_list = np.array(random_forest_list)
            f1_random_forest_list = np.array(f1_random_forest_list)
            acc_dict['RandomForest'] = random_forest_list
            f1_dict['RandomForest'] = f1_random_forest_list
        if one_nn:
            one_nn_list = np.array(one_nn_list)
            f1_one_nn_list = np.array(f1_one_nn_list)
            acc_dict['1-NN'] = one_nn_list
            f1_dict['1-NN'] = f1_one_nn_list
        if three_nn:
            three_nn_list = np.array(three_nn_list)
            f1_three_nn_list = np.array(f1_three_nn_list)
            acc_dict['3-NN'] = three_nn_list
            f1_dict['3-NN'] = f1_three_nn_list
        if five_nn:
            five_nn_list = np.array(five_nn_list)
            f1_five_nn_list = np.array(f1_five_nn_list)
            acc_dict['5-NN'] = five_nn_list
            f1_dict['5-NN'] = f1_five_nn_list
        if log_cls:
            log_reg_list = np.array(log_reg_list)
            f1_log_reg_list = np.array(f1_log_reg_list)
            acc_dict['LogRegression'] = log_reg_list
            f1_dict['LogRegression'] = f1_log_reg_list
        if svm_cls:
            svm_list = np.array(svm_list)
            f1_svm_list = np.array(f1_svm_list)
            acc_dict['SVM'] = svm_list
            f1_dict['SVM'] = f1_svm_list

        return acc_dict, f1_dict

def print_dictionaries_cross_validate(dict_acc, dict_f1, number_of_split):
    print(f"Printing values for n=={number_of_split}")
    for key in dict_acc.keys():
        print(f"Average acc,           for {key}: {dict_acc[key].mean(axis=0)}")
        print(f"Average F1 (Weighted), for {key}: {dict_f1[key].mean(axis=0)}")

cross_val_acc_dict_7, cross_val_f1_dict_7 = cross_validate(7)
cross_val_acc_dict_5, cross_val_f1_dict_5 = cross_validate(5)
cross_val_acc_dict_3, cross_val_f1_dict_3 = cross_validate(3)

print_dictionaries_cross_validate(cross_val_acc_dict_7, cross_val_f1_dict_7, 7)
print_dictionaries_cross_validate(cross_val_acc_dict_5, cross_val_f1_dict_5, 5)
print_dictionaries_cross_validate(cross_val_acc_dict_3, cross_val_f1_dict_3, 3)