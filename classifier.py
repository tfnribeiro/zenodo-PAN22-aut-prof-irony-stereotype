from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from pos_counts import *
from count_features import *
from lexical_comp import *
from read_files import *

# Split the Train data into a 20% Test dataset
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.20)

def get_features(dataset, function, label=""):
    list_features = []
    reported_values = set()
    for i in range(len(dataset)):
        tweet_list = dataset[i]
        get_features = function(tweet_list)
        per_cent_complete = round(i/len(dataset),1)
        if per_cent_complete % 0.2 == 0 and per_cent_complete not in reported_values:
            reported_values.add(per_cent_complete)
            print(f"{label} Processing {per_cent_complete*100}% complete for features: {function.__name__}")
        list_features.append(get_features)
    print(f"{label} Processing for features: {function.__name__}, is complete!")
    return np.array(list_features)

pos_features = get_features(X_train_all, pos_counts, "Train")

count_features = get_features(X_train_all, author_style_counts, "Train")

lix_features = get_features(X_train_all, lix_score, "Train")

emoji_features = get_features(X_train_all, emoji_embeds, "Train")

emoji_pca = PCA(n_components=20)
emoji_pca.fit(emoji_features)
print("Emoji Explained VAR:", emoji_pca.explained_variance_ratio_, "Total VAR:", emoji_pca.explained_variance_ratio_.sum())

print(emoji_features.shape)

X_train_features = np.concatenate((count_features,pos_features,lix_features,emoji_pca.transform(emoji_features)), axis=1)

pos_features = get_features(X_test_all, pos_counts, "Test")

count_features = get_features(X_test_all, author_style_counts, "Test")

lix_features = get_features(X_test_all, lix_score, "Test")

emoji_features = get_features(X_test_all, emoji_embeds, "Test")

print(emoji_features.shape)

X_test_features = np.concatenate((count_features,pos_features,lix_features, emoji_pca.transform(emoji_features)), axis=1)

random_forest_list = []
one_nn_list = []
three_nn_list = []
five_nn_list = []

# KFold validation to pick the best classifier
kf = KFold(n_splits=3)

for i, (train_index, test_index) in enumerate(kf.split(X_train_features)):
    X_train, X_test = X_train_features[train_index,:], X_train_features[test_index,:]
    y_train, y_test = y_train_all[train_index], y_train_all[test_index]
    
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
    print(f"5-NN acc (Train): {acc_train:.4f}")
    print(f"5-NN (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    three_nn_list.append((acc_train,acc_test))
    print(f"3-NN acc (Train): {acc_train:.4f}")
    print(f"3-NN (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    one_nn_list.append((acc_train,acc_test))
    print(f"1-NN acc (Train): {acc_train:.4f}")
    print(f"1-NN (Test): {acc_test:.4f}")

random_forest_list = np.array(random_forest_list)
one_nn_list = np.array(one_nn_list)
three_nn_list = np.array(three_nn_list)
five_nn_list = np.array(five_nn_list)

print(random_forest_list)
print(one_nn_list)
print(three_nn_list)
print(five_nn_list)

clf = RandomForestClassifier()
clf.fit(X_train_features, y_train_all)
y_train_pred = clf.predict(X_train_features)
y_test_pred = clf.predict(X_test_features)
acc_train = np.sum(y_train_pred == y_train_all) / len(y_train_all)
acc_test = np.sum(y_test_pred == y_test_all) / len(y_test_all)

print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")
# Acc: 82% - 90% Depending on the split

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train_features, y_train_all)
print("Log. Reg:", pipe.score(X_test_features, y_test_all))
print(pipe.get_params()['logisticregression'].coef_)

print("Train NI Averages: ")
print(X_train_features[y_train_all=="NI",:].mean(axis=0))
print("Train I Averages: ")
print(X_train_features[y_train_all=="I",:].mean(axis=0))

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