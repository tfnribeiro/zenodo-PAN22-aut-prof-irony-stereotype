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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
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


X_train_n200, X_test_n200, y_train_n200, y_test_n200 = train_test_split(X, y, test_size=0.2)

# Split the Train data into a 20% Test dataset
X_new = []
y_new = []
split_size = 100


for tweet_author_i in range(len(X_train_n200)):
    start = 0
    tweet_author = X_train_n200[tweet_author_i]
    author_label = y_train_n200[tweet_author_i]
    np.random.shuffle(tweet_author)
    for end in range(split_size,len(tweet_author)+1,split_size):
        X_new.append(tweet_author[start:end])
        y_new.append(author_label)
        start = end

X_new = np.array(X_new)
y_new = np.array(y_new)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_new, y_new, test_size=0.1)

#Cache the Values for the test set
#pos
REGEN_FEATURES = True

if not REGEN_FEATURES and os.path.isfile("pos_features.csv"):
    pos_features = np.loadtxt("pos_features.csv", delimiter=",")
else:
    pos_features = get_features(X_train_all, pos_counts, "All Data")
    np.savetxt("pos_features.csv", pos_features, delimiter=",", fmt='%f')

#author style
if not REGEN_FEATURES and os.path.isfile("author_style_counts.csv"):
    print(get_author_style_labels())
    count_features = np.loadtxt("author_style_counts.csv", delimiter=",")
else:
    count_features = get_features(X_train_all, author_style_counts, "All Data")
    np.savetxt("author_style_counts.csv", count_features, delimiter=",", fmt='%f')    

#lix
if not REGEN_FEATURES and os.path.isfile("lix_score.csv"):
    lix_features = np.loadtxt("lix_score.csv", delimiter=",").reshape((-1,1))
else:
    lix_features = get_features(X_train_all, lix_score, "All Data")
    np.savetxt("lix_score.csv", lix_features, delimiter=",", fmt='%f')

#punctuation
if not REGEN_FEATURES and os.path.isfile("punct_score.csv"):
    punct_features = np.loadtxt("punct_score.csv", delimiter=",")
else:
    punct_features = get_features(X_train_all, count_punctuation, "All Data")
    np.savetxt("punct_score.csv", punct_features, delimiter=",", fmt='%f')

#seperated pronunciation
if not REGEN_FEATURES and os.path.isfile("sep_punct_score.csv"):
    sep_punct_features = np.loadtxt("sep_punct_score.csv", delimiter=",")
else:
    sep_punct_features = get_features(X_train_all, seperated_punctuation, "All Data")
    np.savetxt("sep_punct_score.csv", sep_punct_features, delimiter=",", fmt='%f')

#missspelling
if not REGEN_FEATURES and os.path.isfile("misspelled.csv"):
    miss_features = np.loadtxt("misspelled.csv", delimiter=",").reshape((-1,1))
else:
    miss_features = get_features(X_train_all, misspelled, "All Data").reshape((-1,1))
    np.savetxt("misspelled.csv", miss_features, delimiter=",", fmt='%f')

#emoji features
if not REGEN_FEATURES and os.path.isfile("emoji_features.csv"):
    emoji_features = np.loadtxt("emoji_features.csv", delimiter=",")
else:
    emoji_features = get_features(X_train_all, emoji_embeds, "All Data")
    np.savetxt("emoji_features.csv",  emoji_features, delimiter=",", fmt='%f')
    
#sentence polarity
if not REGEN_FEATURES and os.path.isfile("get_sent_polarity.csv"):
    sent_features = np.loadtxt("get_sent_polarity.csv", delimiter=",")
else:
    sent_features = get_features(X_train_all, get_sent_polarity, "All Data")
    np.savetxt("get_sent_polarity.csv", sent_features, delimiter=",", fmt='%f')

#profanity
if not REGEN_FEATURES and os.path.isfile("profanity_counts.csv"):
    profanity_features = np.loadtxt("profanity_counts.csv", delimiter=",")
    
else:
    profanity_features = get_features(X_train_all, profanity_embeds, "All Data")
    np.savetxt("profanity_counts.csv",  profanity_features, delimiter=",", fmt='%f')

def predict(list_authors, classifier):
    pos_features = get_features(list_authors, pos_counts, "Individual Predict", supress_print=True)
    count_features = get_features(list_authors, author_style_counts, "Individual Predict", supress_print=True)
    lix_features = get_features(list_authors, lix_score, "Individual Predict", supress_print=True)
    sent_features = get_features(list_authors, get_sent_polarity, "Individual Predict", supress_print=True)
    sep_punct_features = get_features(list_authors, seperated_punctuation, "Individual Predict", supress_print=True)
    emoji_features = get_features(list_authors, emoji_embeds, "Individual Predict", supress_print=True)
    profanity_features = get_features(list_authors, profanity_embeds, "Individual Predict", supress_print=True)

    emoji_features = emoji_pca.transform(emoji_features)
    profanity_features = profanity_pca.transform(profanity_features)

    x = np.concatenate((count_features, pos_features, lix_features, emoji_features, sent_features, sep_punct_features, profanity_features), axis=1)
    return classifier.predict(x), classifier.predict_proba(x)

#emoji_pca = PCA(n_components=5) 
#emoji_features = emoji_pca.fit_transform(emoji_features)
#profanity_pca = SparsePCA(n_components=10)
#profanity_features = profanity_pca.fit_transform(profanity_features)
#
emoji_pca_n = 5 
profanity_components = 10

emoji_pca = SparsePCA(n_components=emoji_pca_n)
profanity_pca = SparsePCA(n_components=profanity_components) 

print("Performing Emoji-PCA")
emoji_features_train = emoji_pca.fit_transform(emoji_features)
emoji_features_test = emoji_pca.transform(emoji_features)
print("Performing Profanity-PCA")
profanity_features_train = profanity_pca.fit_transform(profanity_features)
profanity_features_test = profanity_pca.transform(profanity_features)

X_train_all_features =  np.concatenate((count_features,pos_features, lix_features, 
        emoji_features_train, sent_features, sep_punct_features, profanity_features_train), axis=1)#
#feature_df = pd.DataFrame(np.concatenate((USERCODE_X.reshape((-1,1)),X_train_features,y.reshape((-1,1))),axis=1), columns=["input_file", "auth_vocabsize","type_token_rt","avg_author_word_length",
#"avg_tweet_length","avg_author_hashtag_count","avg_author_usertag_count","avg_author_urltag_count",
#"author_avg_emoji","avg_capital_lower_ratio","ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",
#"PUNCT","UNK","LiXScore", "emoji_pca_1", "emoji_pca_2", "emoji_pca_3", "emoji_pca_4", "emoji_pca_5", "pos", "neut", "neg", "compound",
#"punct_normal_features_count", "punct_weird_features_count", "profanity_pca_1", "profanity_pca_2", "profanity_pca_3", "profanity_pca_4", "profanity_pca_5", "profanity_pca_6",
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
kf = KFold(n_splits=5)

print(f"Performing Cross-Validation...")
for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_train_all_features))):
    print("Train: ", len(train_index), "Test: ", len(test_index))
    
    print("Performing Emoji-PCA")
    emoji_features_train = emoji_pca.fit_transform(emoji_features[train_index,:])
    emoji_features_test = emoji_pca.transform(emoji_features[test_index,:])

    print("Performing Profanity-PCA")
    profanity_features_train = profanity_pca.fit_transform(profanity_features[train_index,:])
    profanity_features_test = profanity_pca.transform(profanity_features[test_index,:])

    X_train=  np.concatenate((count_features[train_index,:],pos_features[train_index,:], lix_features[train_index,:], 
        emoji_features_train, sent_features[train_index,:], sep_punct_features[train_index,:], profanity_features_train), axis=1)
    
    X_test = np.concatenate((count_features[test_index,:],pos_features[test_index,:], lix_features[test_index,:], 
        emoji_features_test, sent_features[test_index,:], sep_punct_features[test_index,:], profanity_features_test), axis=1)

    y_train, y_test = y_train_all[train_index], y_train_all[test_index]

    print(X_train.shape)

    #clf = ExtraTreesClassifier(n_estimators=50)
    #clf = clf.fit(X_train, y_train)
    #model = SelectFromModel(clf, prefit=True)
    #X_train = model.transform(X_train)
    #X_test = model.transform(X_test)
    #print(X_train.shape) 
    
    
    ratio_train_i = (y_train == "I").sum()/len(y_train)
    ratio_train_ni = (y_train == "NI").sum()/len(y_train)

    ratio_test_i = (y_test == "I").sum()/len(y_test)
    ratio_test_ni = (y_test == "NI").sum()/len(y_test)
    print(f"Train Ratio of Labels (split: {i+1}): I: {ratio_train_i} | NI:{ratio_train_ni}")
    print(f"Test Ratio of Labels (split: {i+1}): I: {ratio_test_i} | NI:{ratio_test_ni}")
    #X_train, X_test, y_train, y_test = train_test_split(list_features, y, test_size=0.3)

    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
    y_train_pred = clf_rfc.predict(X_train)
    y_test_pred = clf_rfc.predict(X_test)

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

clf_rfc = RandomForestClassifier()
clf_rfc.fit(X_train_all_features, y_train_all)

y_test_pred, _ = predict(X_test_all, clf_rfc)
print("W. F1-Score (Unseen data Splitted): ", f1_score(y_test_pred, y_test_all, average='weighted'))

def test_predictions(classfier, n_sentences=50):
    test_sentences = n_sentences
    predictions = []
    labels = []
    for author_i, label in enumerate(y_test_all):
        start = 0
        current_author = X_test_all[author_i]
        np.random.shuffle(current_author)
        for end in np.arange(test_sentences, len(current_author)+1, test_sentences):
            sentence_sample = current_author[start:end]
            prediction, probability = predict([sentence_sample], classfier)
            predictions.append(prediction)
            labels.append(label)
            start = end
        if author_i % 10 == 0:
            #print(f"The prediction {i} (n=={n_sentences}): {probability}, P:{prediction} | L:{label}")
            current_f1_score = f1_score(np.array(predictions), np.array(labels),average='weighted')
            print(f"The prediction acc (n=={n_sentences}) after {author_i+1} authors: {current_f1_score}")
    predictions = np.array(predictions)
    labels = np.array(labels)
    current_f1_score = f1_score(predictions, labels, average='weighted')
    print(f"Final F1-Score (n=={n_sentences}): {current_f1_score}")
    return predictions==labels, predictions, labels

acc_list_10, pred, labels = test_predictions(clf_rfc, 10)
acc_list_20, pred, labels = test_predictions(clf_rfc, 20)
acc_list_50, pred, labels = test_predictions(clf_rfc)
acc_list_75, pred, labels = test_predictions(clf_rfc, 75)

#print(acc_list_75.sum()/len(acc_list_75))

all_data_pred, all_data_prob = predict(X_test_n200, clf_rfc)
print("W. F1-Score (Unseen Full): ", f1_score(all_data_pred, y_test_n200,average='weighted'))

def train_predict(train, train_labels, test):
    pos_features = get_features(train, pos_counts, "Individual Predict", supress_print=True)
    count_features = get_features(train, author_style_counts, "Individual Predict", supress_print=True)
    lix_features = get_features(train, lix_score, "Individual Predict", supress_print=True)
    sent_features = get_features(train, get_sent_polarity, "Individual Predict", supress_print=True)
    sep_punct_features = get_features(train, count_punctuation, "Individual Predict", supress_print=True)
    miss_features = get_features(train, misspelled, "Individual Predict", supress_print=True).reshape((-1,1))
    emoji_features = get_features(train, emoji_embeds, "Individual Predict", supress_print=True)
    profanity_features = get_features(train, profanity_embeds, "Individual Predict", supress_print=True)
    emoji_pca_n = 5 
    profanity_components = 10
    emoji_pca = SparsePCA(n_components=emoji_pca_n)
    profanity_pca = SparsePCA(n_components=profanity_components) 
    emoji_features = emoji_pca.fit_transform(emoji_features)
    profanity_features = profanity_pca.fit_transform(profanity_features)
    x_train = np.concatenate((count_features, pos_features, lix_features, emoji_features, sent_features, sep_punct_features, profanity_features), axis=1)

    classifier = RandomForestClassifier()
    classifier.fit(x_train, train_labels)

    pos_features = get_features(test, pos_counts, "Individual Predict", supress_print=True)
    count_features = get_features(test, author_style_counts, "Individual Predict", supress_print=True)
    lix_features = get_features(test, lix_score, "Individual Predict", supress_print=True)
    sent_features = get_features(test, get_sent_polarity, "Individual Predict", supress_print=True)
    sep_punct_features = get_features(test, count_punctuation, "Individual Predict", supress_print=True)
    miss_features = get_features(test, misspelled, "Individual Predict", supress_print=True).reshape((-1,1))
    emoji_features = get_features(test, emoji_embeds, "Individual Predict", supress_print=True)
    profanity_features = get_features(test, profanity_embeds, "Individual Predict", supress_print=True)
    emoji_features = emoji_pca.transform(emoji_features)
    profanity_features = profanity_pca.transform(profanity_features)

    x_test = np.concatenate((count_features, pos_features, lix_features, emoji_features, sent_features, sep_punct_features, profanity_features), axis=1)

    return classifier.predict(x_test), classifier.predict_proba(x_test)

all_data_pred, all_data_prob = train_predict(X_train_n200, y_train_n200, X_test_n200)
print("W. F1-Score (Unseen Full), Trained on full: ", f1_score(all_data_pred, y_test_n200,average='weighted'))
"""
"""