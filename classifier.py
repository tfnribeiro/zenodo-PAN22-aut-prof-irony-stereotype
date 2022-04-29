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
from misspelings import *
import pandas as pd
import os 
np.random.seed(0)

def get_features(dataset, function, label="", report_per_cent=50):
    list_features = []
    reported_values = set()
    for i in range(len(dataset)):
        tweet_list = dataset[i]
        get_features = function(tweet_list)
        per_cent_complete = round(i/len(dataset),2)*100
        if  per_cent_complete % report_per_cent == 0 and per_cent_complete not in reported_values:
            reported_values.add(per_cent_complete)
            print(f"{label} Processing {per_cent_complete}% complete for features: {function.__name__}")
        list_features.append(get_features)
    print(f"{label} Processing for features: {function.__name__}, is complete!")
    return np.array(list_features)

# Split the Train data into a 20% Test dataset
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.25)

#Cache the Values for the test set
if os.path.isfile("pos_features.csv"):
    pos_features = np.loadtxt("pos_features.csv", delimiter=",")
else:
    pos_features = get_features(X, pos_counts, "All Data")
    np.savetxt("pos_features.csv", pos_features, delimiter=",", fmt='%f')

if os.path.isfile("author_style_counts.csv"):
    count_features = np.loadtxt("author_style_counts.csv", delimiter=",")
else:
    count_features = get_features(X, author_style_counts, "All Data")
    np.savetxt("author_style_counts.csv", count_features, delimiter=",", fmt='%f')
    
if os.path.isfile("punct_score.csv"):
    punct_features = np.loadtxt("punct_score.csv", delimiter=",")
else:
    punct_features = get_features(X, count_punctuation, "All Data")
    np.savetxt("punct_score.csv", punct_features, delimiter=",", fmt='%f')
    
if os.path.isfile("misspelled.csv"):
    miss_features = np.loadtxt("misspelled.csv", delimiter=",").reshape((-1,1))
else:
    miss_features = get_features(X, misspelled, "All Data").reshape((-1,1))
    np.savetxt("misspelled.csv", miss_features, delimiter=",", fmt='%f')

if os.path.isfile("lix_score.csv"):
    lix_features = np.loadtxt("lix_score.csv", delimiter=",").reshape((-1,1))
else:
    lix_features = get_features(X, lix_score, "All Data")
    np.savetxt("lix_score.csv", lix_features, delimiter=",", fmt='%f')

if os.path.isfile("punct_score.csv"):
    punct_features = np.loadtxt("punct_score.csv", delimiter=",")
else:
    punct_features = get_features(X, count_punctuation, "All Data")
    np.savetxt("punct_score.csv", punct_features, delimiter=",", fmt='%f')
    
if os.path.isfile("misspelled.csv"):
    miss_features = np.loadtxt("misspelled.csv", delimiter=",").reshape((-1,1))
else:
    miss_features = get_features(X, misspelled, "All Data").reshape((-1,1))
    np.savetxt("misspelled.csv", miss_features, delimiter=",", fmt='%f')


if os.path.isfile("emoji_features.csv"):
    emoji_features = np.loadtxt("emoji_features.csv", delimiter=",")
else:
    emoji_features = get_features(X, emoji_embeds, "All Data")
    np.savetxt("emoji_features.csv",  emoji_features, delimiter=",", fmt='%f')
    

if os.path.isfile("get_sent_polarity.csv"):
    sent_features = np.loadtxt("get_sent_polarity.csv", delimiter=",")
else:
    sent_features = get_features(X, get_sent_polarity, "All Data")
    np.savetxt("get_sent_polarity.csv", sent_features, delimiter=",", fmt='%f')


if os.path.isfile("profanity_counts.csv"):
    profanity_features = np.loadtxt("profanity_counts.csv", delimiter=",")
    
else:
    profanity_features = get_features(X, profanity_embeds, "All Data")
    np.savetxt("profanity_counts.csv",  profanity_features, delimiter=",", fmt='%f')

#X_train_features = np.concatenate((count_features,pos_features, lix_features, emoji_features, sent_features, punct_features, miss_features, profanity_features), axis=1)

#feature_df = pd.DataFrame(np.concatenate((USERCODE_X.reshape((-1,1)),X_train_features,y.reshape((-1,1))),axis=1), columns=["input_file", "auth_vocabsize","type_token_rt","author_word_length_avg",
#"avg_tweet_length","author_hashtag_count","author_usertag_count","author_urltag_count",
#"author_avg_emoji","avg_capital_lower_ratio","ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",
#"PUNCT","UNK","LiXScore", "emoji_pca_1", "emoji_pca_2", "emoji_pca_3", "emoji_pca_4", "emoji_pca_5", "pos", "neut", "neg", "compound",
#"punct_normal_features_count", "punct_weird_features_count", "missspelled_features", "profanity_pca_1", "profanity_pca_2", "profanity_pca_3", "profanity_pca_4", "profanity_pca_5", "profanity_pca_6",
#"profanity_pca_7", "profanity_pca_8","profanity_pca_9","profanity_pca_10", "label"])
#feature_df.to_csv("pd_X_features.csv")

random_forest_list = []
one_nn_list = []
three_nn_list = []
five_nn_list = [] 
log_reg_list = []
svm_list = []

f1_random_forest_list = []
f1_one_nn_list = []
f1_three_nn_list = []
f1_five_nn_list = []
f1_log_reg_list = []
f1_svm_list = []
# KFold validation to pick the best classifier
kf = KFold(n_splits=4)

emoji_pca_n = 5
profanity_components = 10

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("Train: ", len(train_index), "Test: ", len(test_index))
    emoji_pca = PCA(n_components=emoji_pca_n) 
    emoji_features_train = emoji_pca.fit_transform(emoji_features[train_index,:])
    emoji_features_test = emoji_pca.transform(emoji_features[test_index,:])

    profanity_pca = SparsePCA(n_components=profanity_components)
    profanity_features_train = profanity_pca.fit_transform(emoji_features[train_index,:])
    profanity_features_test = profanity_pca.transform(emoji_features[test_index,:])

    X_train=  np.concatenate((count_features[train_index,:],pos_features[train_index,:], lix_features[train_index,:], 
        emoji_features_train, sent_features[train_index,:], punct_features[train_index,:], miss_features[train_index,:], profanity_features_train), axis=1)
    
    X_test = np.concatenate((count_features[test_index,:],pos_features[test_index,:], lix_features[test_index,:], 
        emoji_features_test, sent_features[test_index,:], punct_features[test_index,:], miss_features[test_index,:], profanity_features_test), axis=1)
    
    
    y_train, y_test = y[train_index], y[test_index]
    
    ratio_train_i = (y_train == "I").sum()/len(y_train)
    ratio_train_ni = (y_train == "NI").sum()/len(y_train)

    ratio_test_i = (y_test == "I").sum()/len(y_test)
    ratio_test_ni = (y_test == "NI").sum()/len(y_test)
    print(f"Train Ratio of Labels (split: {i+1}): I: {ratio_train_i} | NI:{ratio_train_ni}")
    print(f"Test Ratio of Labels (split: {i+1}): I: {ratio_test_i} | NI:{ratio_test_ni}")
    #X_train, X_test, y_train, y_test = train_test_split(list_features, y, test_size=0.3)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    f1_random_forest_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    random_forest_list.append((acc_train,acc_test))
    print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
    print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    five_nn_list.append((acc_train,acc_test))
    f1_five_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    print(f"5-NN acc (Train): {acc_train:.4f}")
    print(f"5-NN (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    three_nn_list.append((acc_train,acc_test))
    f1_three_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    print(f"3-NN acc (Train): {acc_train:.4f}")
    print(f"3-NN (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    one_nn_list.append((acc_train,acc_test))
    f1_one_nn_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    print(f"1-NN acc (Train): {acc_train:.4f}")
    print(f"1-NN (Test): {acc_test:.4f}")
    
    pipe = make_pipeline(StandardScaler(), svm.SVC(gamma="auto"))
    pipe.fit(X_train, y_train)
    acc_train = pipe.score(X_train, y_train)
    acc_test = pipe.score(X_test, y_test)
    print("SVM:", acc_test)
    svm_list.append((acc_train,acc_test))
    f1_svm_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    
    #print("Train NI Averages: ")
    #print(X_train[y_train=="NI",:].mean(axis=0))
    #print("Train I Averages: ")
    #print(X_train[y_train=="I",:].mean(axis=0))

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)
    print("Log. Reg:", pipe.score(X_test, y_test))
    acc_train = pipe.score(X_train, y_train)
    acc_test = pipe.score(X_test, y_test)
    log_reg_list.append((acc_train,acc_test))
    f1_log_reg_list.append(f1_score(y_test, y_test_pred,average='weighted'))
    

random_forest_list = np.array(random_forest_list)
one_nn_list = np.array(one_nn_list)
three_nn_list = np.array(three_nn_list)
five_nn_list = np.array(five_nn_list)
log_reg_list = np.array(log_reg_list)

f1_random_forest_list = np.array(f1_random_forest_list)
f1_one_nn_list = np.array(f1_one_nn_list)
f1_three_nn_list = np.array(f1_three_nn_list)
f1_five_nn_list = np.array(f1_five_nn_list)
f1_log_reg_list = np.array(f1_log_reg_list)
f1_svm_list = np.array(f1_svm_list)

print("Random Forest Averages: ", random_forest_list.mean(axis=0))
print("1-NN Averages: ", one_nn_list.mean(axis=0))
print("3-NN Averages: ", three_nn_list.mean(axis=0))
print("5-NN Averages: ", five_nn_list.mean(axis=0))
print("Log Reg Averages: ", log_reg_list.mean(axis=0))

print("Random Forest (F1-Score) W.Averages: ", f1_random_forest_list.mean(axis=0))
print("1-NN (F1-Score) W.Averages: ", f1_one_nn_list.mean(axis=0))
print("3-NN (F1-Score) W.Averages: ", f1_three_nn_list.mean(axis=0))
print("5-NN (F1-Score) W.Averages: ", f1_five_nn_list.mean(axis=0))
print("Log Reg (F1-Score) W.Averages: ", f1_log_reg_list.mean(axis=0))
print("SVM (F1-Score) W.Averages: ", f1_svm_list.mean(axis=0))

#np.savetxt("covariance.tsf", np.corrcoef(X_train_features, rowvar=False), delimiter="\t", fmt='%f')
#clf = RandomForestClassifier()
#clf.fit(X_train_features, y_train_all)
#y_train_pred = clf.predict(X_train_features)
#y_test_pred = clf.predict(X_test_features)
#acc_train = np.sum(y_train_pred == y_train_all) / len(y_train_all)
#acc_test = np.sum(y_test_pred == y_test_all) / len(y_test_all)
#
#print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
#print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")
## Acc: 82% - 90% Depending on the split
#
#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#pipe.fit(X_train_features, y_train_all)
#print("Log. Reg:", pipe.score(X_test_features, y_test_all))
#print(pipe.get_params()['logisticregression'].coef_)
#
#print("Train NI Averages: ")
#print(X_train_features[y_train_all=="NI",:].mean(axis=0))
#print("Train I Averages: ")
#print(X_train_features[y_train_all=="I",:].mean(axis=0))

"""

Weights from Log Regression gives us the "predictive power":
[[ 0.83384098  0.43150988 -0.31436612  1.12271948 -0.71178175 -0.61877302
  -0.17990646 -0.13489198 -0.13489198  0.3576268   0.48496003 -0.10197374
  -0.24532996  0.12368179 -1.235854    0.29782858  0.45247568 -0.27540978
   1.25189261 -0.46820761 -0.65596446 -0.33554537  1.21193532]]

[auth_vocabsize, type_token_rt, author_word_length_avg, avg_tweet_length, author_hashtag_count, author_usertag_count, 
 author_urltag_count, author_avg_emoji, avg_capital_lower_ratio, ADJ, ADP, ADV , 
 CONJ, DET, NOUN, NUM, PRT, PRON, 
 VERB, PUNCT, UNK, LiXScore]

"""