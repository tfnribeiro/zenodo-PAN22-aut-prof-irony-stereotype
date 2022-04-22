from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from pos_counts import *
from count_features import *

# Split the Train data into a 20% Test dataset
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.2)

list_features = []
for i in range(len(X_train_all)):
    tweet_list = X_train_all[i]
    get_features = pos_counts(tweet_list)
    print(y[i], get_features)
    list_features.append(get_features)

pos_features = np.array(list_features)

list_features = []
for i in range(len(X_train_all)):
    tweet_list = X_train_all[i]
    get_features = author_style_counts(tweet_list)
    print(y[i], get_features)
    list_features.append(get_features)

count_featues = np.array(list_features)

X_train_features = np.hstack((count_featues,pos_features))

list_features = []
for i in range(len(X_test_all)):
    tweet_list = X_test_all[i]
    get_features = pos_counts(tweet_list)
    print(y[i], get_features)
    list_features.append(get_features)

pos_features = np.array(list_features)

list_features = []
for i in range(len(X_test_all)):
    tweet_list = X_test_all[i]
    get_features = author_style_counts(tweet_list)
    print(y[i], get_features)
    list_features.append(get_features)

count_featues = np.array(list_features)

X_test_features = np.hstack((count_featues,pos_features))

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
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train_features)
y_test_pred = clf.predict(X_test_features)
acc_train = np.sum(y_train_pred == y_train_all) / len(y_train_all)
acc_test = np.sum(y_test_pred == y_test_all) / len(y_test_all)

print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")
# Acc: 82% - 90% Depending on the split

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
print("Log. Reg:", pipe.score(X_test, y_test))
print(pipe.get_params()['logisticregression'].coef_)


"""

Weights from Log Regression gives us the "predictive power":
[[ 0.8165839   0.15800493 -0.10996453  1.15362878 -0.71378012 -0.69999862
  -0.00029913 -0.00029913  0.23670419 -0.25793926  0.24235472  0.64781405
  -1.39277626  1.50101159  0.45641181  0.04703658  0.93722263 -0.78494383
  -0.93381287 -0.61578778]]
[auth_vocabsize, type_token_rt, author_word_length_avg, avg_tweet_length, author_hashtag_count, author_usertag_count, 
author_total_emoji, author_avg_emoji, ADJ, ADP , ADV , CONJ,
DET, NOUN, NUM, PRT, PRON, VERB,
PUNCT, UNK]

"""