from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from read_files import *
import classifier_methods as cm


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
    X_features, _, _, _, _, _, _ = cm.get_features_train(X[:302])
    scaler = MinMaxScaler()
    fited = scaler.fit_transform(X_features)
    
    feature_names = [ 
    "ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB","PUNCT","UNK",
    "auth_vocabsize","type_token_rt","avg_author_word_length","avg_tweet_length","avg_author_hashtag_count",
    "avg_author_usertag_count","avg_author_urltag_count","author_avg_emoji","avg_capital_lower_ratio",
    "pos", "neut", "neg", "compound",
    "mult-ex", "mult-qu", "mult-pe", "quote", "ex", "qu", "pe",
    "LiXScore",
    "emoji_pca_1", "emoji_pca_2", "emoji_pca_3", "emoji_pca_4", "emoji_pca_5",
    "profanity_pca_1", "profanity_pca_2", "profanity_pca_3", "profanity_pca_4", "profanity_pca_5", 
    "profanity_pca_6","profanity_pca_7", "profanity_pca_8","profanity_pca_9","profanity_pca_10",
    "word_pca1","word_pca2","word_pca3", "word_pca4", "word_pca5", "word_pca6", "word_pca7", "word_pca8",
    "word_pca9", "word_pca10", "word_pca11", "word_pca12", "word_pca13", "word_pca14","word_pca15","word_pca16",
    "word_pca17", "word_pca18" , "word_pca19", "word_pca20"
    ]

    features = most_important_features(X_features, feature_names)
    print(features)
