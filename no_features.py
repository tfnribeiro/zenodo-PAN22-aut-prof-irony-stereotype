# Feature Extraction with PCA
from tkinter import Y
from classifier_methods import get_features_train
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
import matplotlib.pyplot as plt
from read_files import *
from pos_counts import *
from count_features import *
from lexical_comp import *
from sent_polarity import *
from read_files import *
from punctuation import *
from tqdm import tqdm
import classifier_methods as cm

'''
pos_features = np.loadtxt("pos_features.csv", delimiter=",")
count_features = np.loadtxt("author_style_counts.csv", delimiter=",")
sent_features = np.loadtxt("./get_sent_polarity.csv", delimiter=",")
sep_punct_features = np.loadtxt("./sep_punct_score.csv", delimiter=",")
lix_features = np.loadtxt("lix_score.csv", delimiter=",")
emoji_tfidf = fit_emoji_embeds_tfidf(X)
emoji_tfidf_features = get_features(X, emoji_tfidf.tf_idf, "Emoji TF_IDF")

profanity_tfidf = fit_profanity_embeds_tfidf(X)
profanity_tfidf_features = get_features(X, profanity_tfidf.tf_idf, "Profanity TF_IDF")

words_tfidf = fit_word_embeds_tfidf(X)
words_tfidf_features = get_features(X, words_tfidf.tf_idf, "Words TF_IDF")

emoji_pca_n =5
profanity_pca_n = 10
word_pca_n = 20

emoji_pca = PCA(n_components=emoji_pca_n)
profanity_pca = PCA(n_components=profanity_pca_n)
word_pca = PCA(n_components=word_pca_n)



emoji_features_train = emoji_pca.fit_transform(emoji_tfidf_features)
profanity_features_train = profanity_pca.fit_transform(profanity_tfidf_features)
word_features_train = word_pca.fit_transform(words_tfidf_features)
#pos_features, count_features, sent_features, sep_punct_features, lix_features, emoji_features_train, 
# profanity_features_train, word_features_train
print(pos_features.shape)
print(count_features.shape) 
print(sent_features.shape) 
print(sep_punct_features.shape) 
print(lix_features.reshape((-1,1)).shape)
print(emoji_features_train.shape)
print(profanity_features_train.shape)
print(word_features_train.shape)
print(y.shape)
'''
X_features, _, _, _, _, _, _ = cm.get_features_train(X[:302])
y[y == 'I'] = 1
y[y == 'NI'] = 0
#X_features = np.concatenate((pos_features, count_features, sent_features, sep_punct_features, lix_features.reshape((-1,1)), emoji_features_train, profanity_features_train, word_features_train), axis=1) #, y.reshape(-1,1)


# load data
'''
feature_df = pd.DataFrame(X_features, 
columns=[ "ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",
"PUNCT","UNK","auth_vocabsize","type_token_rt","avg_author_word_length",
"avg_tweet_length","avg_author_hashtag_count","avg_author_usertag_count","avg_author_urltag_count",
"author_avg_emoji","avg_capital_lower_ratio", "pos", "neut", "neg", "compound",
"mult-ex", "mult-qu", "mult-pe", "quote", "ex", "qu", "pe","LiXScore",
"emoji_pca_1", "emoji_pca_2", "emoji_pca_3", "emoji_pca_4", "emoji_pca_5",
"profanity_pca_1", "profanity_pca_2", "profanity_pca_3", "profanity_pca_4", "profanity_pca_5", "profanity_pca_6",
"profanity_pca_7", "profanity_pca_8","profanity_pca_9","profanity_pca_10",
"word_pca1","word_pca2","word_pca3", "word_pca4", "word_pca5", "word_pca6", "word_pca7", "word_pca8",
"word_pca9", "word_pca10", "word_pca11", "word_pca12", "word_pca13", "word_pca14","word_pca15"
"word_pca16", "word_pca17", "word_pca18" , "word_pca19", "word_pca20", "label"])
'''




'''
df = feature_df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['input_file'],axis=1)
df['label'][df.label == 'I'] = 1
df['label'][df.label == 'NI'] = 0
print(df.head())
array = df.values
Y = df['label']
'''


#X = df.drop(['label'], axis=1)
#print(X.head())




# feature extraction
n_c = X_features.shape[1]

pca = PCA(n_components=n_c)
fit = pca.fit(X_features)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
#print(fit.components_)
c_var = np.cumsum(fit.explained_variance_ratio_)

print(c_var)
'''
plt.plot(c_var)
plt.grid('on')
plt.xlabel('Comulative of PCs in descending order occupancy, normalised')
plt.ylabel('Accumulated projected variance')
plt.title('Cummulative valriance of norm data')
plt.show()
'''


n_pcs= fit.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(fit.components_[i]).argmax() for i in range(n_pcs)]



print(most_important)
initial_feature_names = [ "ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB","PUNCT","UNK",
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
"word_pca17", "word_pca18" , "word_pca19", "word_pca20"] #, "label"]
# get the names
print(X_features.shape)
print(len(initial_feature_names))
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)


'''
['pe', 'mult-pe', 'ex', 'avg_tweet_length', 'quote', 'qu', 'mult-ex', 'mult-qu', 'LiXScore', 'NOUN',
'word_pca2', 'word_pca3', 'auth_vocabsize', 'word_pca6', 'word_pca7', 'avg_author_usertag_count',
'word_pca8', 'word_pca9', 'word_pca13', 'word_pca12', 'word_pca15', 'word_pca16', 'word_pca18',
'word_pca20', 'word_pca17', 'word_pca10', 'word_pca17', 'avg_author_usertag_count', 'ADJ',
'word_pca19', 'word_pca15', 'ADP', 'DET', 'VERB', 'emoji_pca_4', 'emoji_pca_2', 'emoji_pca_3',
'emoji_pca_3', 'avg_author_hashtag_count', 'emoji_pca_4', 'emoji_pca_5', 'emoji_pca_5',
'profanity_pca_1', 'avg_author_urltag_count', 'NUM', 'CONJ', 'profanity_pca_2', 'PRT',
'profanity_pca_2', 'avg_capital_lower_ratio', 'PRT', 'profanity_pca_9', 'profanity_pca_4',
'profanity_pca_5', 'avg_author_word_length', 'profanity_pca_8', 'profanity_pca_10',
'profanity_pca_10', 'profanity_pca_9', 'profanity_pca_7', 'compound', 'profanity_pca_6',
'profanity_pca_4', 'neut', 'UNK', 'type_token_rt', 'pos', 'pos']
'''