from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import classifier_methods as cm
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from utils import *
import time
import pandas as pd


def most_important_features(X_features, initial_feature_names):
    '''
    Takes the features and a list of feature names
    sorting the features by importance via PCA 
    :param: X_features features for each author
    :param: initial_feature_names names of the features
    return: list of feature names sorted by importance
    '''

    # feature extraction
    n_c = X_features.shape[1]

    pca = PCA(n_components=n_c)
    fit = pca.fit(X_features)
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    '''
    c_var = np.cumsum(fit.explained_variance_ratio_)

    print(c_var)
    
    plt.plot(c_var)
    plt.grid('on')
    plt.xlabel('Comulative of PCs in descending order occupancy, normalised')
    plt.ylabel('Accumulated projected variance')
    plt.title('Cummulative valriance of norm data')
    plt.show()
    '''
    
    n_pcs= fit.components_.shape[0]

    # get the index of the most important feature on each component
    ordered = [np.abs(fit.components_[i]).argmax() for i in range(n_pcs)]

    #order the features
    ordered_feature_names = [initial_feature_names[ordered[i]] for i in range(n_pcs)]

    #print(most_important_names)
    return ordered_feature_names


if __name__ == "__main__":
    X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # make_pipeline(StandardScaler(), LogisticRegression()) RandomForestClassifier() make_pipeline(StandardScaler(), svm.SVC(gamma="auto"))
    forest , emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, X_train_features = cm.train_model(X_train, y_train, make_pipeline(RandomForestClassifier()))
    
    X_test_features = cm.get_features_test(X_test,emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf)
    predict = forest.predict(X_test_features)
    print((y_test==predict).sum()/len(y_test))
    start_time = time.time()
    result = permutation_importance(
        forest, X_test_features, y_test, n_repeats=5, n_jobs=5
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    
    # X_features, _, _, _, _, _, _ = cm.get_features_train(X[:302])
    # scaler = MinMaxScaler()
    # fited = scaler.fit_transform(X_features)
    feature_names = [ 
    "ADJ","ADP","ADV","CONJ","DET",#"NOUN",
    "NUM","PRT","PRON","VERB",
    #"PUNCT",
    #"UNK",
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

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    print(get_top_pca_values(emoji_pca,emoji_tfidf,10))
