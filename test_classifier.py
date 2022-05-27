from matplotlib.pyplot import get
from classifier_methods import *
from utils import * 
import json

X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
##x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X)
#
##generate_predictions_output(predictions, USERCODE_X, lang)
# Emoji: 65, Profanity: 95,  Word: 255, 90% Variance (All X, this might not hold for different folds)
# predictions, *rest, classifier = generate_features_train_predict(X_train, y_train, X_test)
# Best Params (ACC) so far - 0.9353594389246055 : Emoji_n:5 | Profanity_n:9 | Word_n:10
# Best Params (ACC) so far - 0.9318527177089422 : Emoji_n:10 | Profanity_n:10 | Word_n:10
# Best Params (ACC) so far - 0.9338245614035088 : Emoji_n:10 | Profanity_n:15 | Word_n:15
# 0.939157894736842 : Emoji_n:4 | Profanity_n:14 | Word_n:20

def tune_params():
    acc_dict, f1_dict, best_e, best_p, best_w = cross_validate_tune_params(X_train, y_train, 5, emoji_pca_dim=np.arange(2,5,2), profanity_pca_dim=np.arange(10,16,2), word_pca_dim=np.arange(15,26,5))

    # create json object from dictionary
    acc_dict_json = dict()
    for classifier in acc_dict.keys():
        acc_dict_json[classifier] = dict()
        for params in acc_dict[classifier].keys():
            acc_dict_json[classifier][str(params)] = acc_dict[classifier][params].mean(axis=0)[1]

    f1_dict_json = dict()
    for classifier in f1_dict.keys():
        f1_dict_json[classifier] = dict()
        for params in f1_dict[classifier].keys():
            f1_dict_json[classifier][str(params)] = f1_dict[classifier][params].mean()

    json_acc = json.dumps(acc_dict_json)
    json_f1 = json.dumps(f1_dict_json)

    with open("acc_search_value.json","w") as f:
        f.write(json_acc)

    with open("f1_search_value.json","w") as f:
        f.write(json_f1)

    predictions, _, prob, classifier = generate_features_train_predict(X_train, y_train, X_test, classifier_class=RandomForestClassifier(), emoji_pca_dim=4, profanity_pca_dim=14, word_pca_dim=20, label="Test", supress_prints_flag=False)

    print("Acc: ", sum(predictions==y_test)/len(y_test))

#acc, f1 = cross_validate(X_train, y_train)
#predictions, _, prob, classifier = generate_features_train_predict(X_train, y_train, X_test, classifier_class=RandomForestClassifier(), emoji_pca_dim=4, profanity_pca_dim=14, #word_pca_dim=20, label="Test", supress_prints_flag=False)
#
#print(f1_score(y_test, predictions))

# x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X)

# top_20_word_tfidf = get_top_idf_values(X, y, words_tfidf)

embed_features = get_features(X_train, tweet_word_embs)
embed_features_test = get_features(X_test, tweet_word_embs)
clf = RandomForestClassifier()
clf.fit(embed_features, y_train)
print((y_test == clf.predict(embed_features_test)).sum()/len(y_test))
 
"""
With Std:
>>> acc['RandomForest'].mean(axis=0)
array([1.        , 0.93154762])
>>> acc['LogRegression'].mean(axis=0)
array([0.97470238, 0.90178571])
>>> acc['SVM'].mean(axis=0)
array([0.97271825, 0.91071429])
>>> (y_test == predictions).sum()/len(y_test)
0.9404761904761905

With Mean:
>>> acc['RandomForest'].mean(axis=0)
array([1.        , 0.92559524])
>>> acc['SVM'].mean(axis=0)
array([0.97172619, 0.9077381 ])
>>> acc['LogRegression'].mean(axis=0)
array([0.97668651, 0.90178571])
>>> (y_test == predictions).sum()/len(y_test)
0.9166666666666666

"""
# emoji_tfidf = fit_emoji_embeds_tfidf(
#     X, authors_document=False)
# emoji_tfidf_features = get_features(
#     X, emoji_tfidf.tf_idf, "Emoji TF_IDF")a
# profanity_tfidf = fit_profanity_embeds_tfidf(
#     X, authors_document=False)
# profanity_tfidf_features = get_features(
#     X, profanity_tfidf.tf_idf, "Profanity TF_IDF")
# words_tfidf = fit_word_embeds_tfidf(
#     X, authors_document=False)
# words_tfidf_features = get_features(
#     X, words_tfidf.tf_idf, "Words TF_IDF")
# 
# for emoji_n in range(5,86,5):
#     emoji_pca = PCA(n_components=emoji_n)
#     emoji_features_train = emoji_pca.fit(emoji_tfidf_features)
#     print(f"N=={emoji_n} Emoji Explained Variance: ", sum(emoji_pca.explained_variance_ratio_))
# 
# for profanity_n in range(10, 106, 5):
#     profanity_pca = PCA(n_components=profanity_n)
#     profanity_features_train = profanity_pca.fit( 
#             profanity_tfidf_features)
#     print(f"N=={profanity_n} Profanity Explained Variance: ", sum(profanity_pca.explained_variance_ratio_))
# 
# for word_n in range(220,261,10):
#     word_pca = PCA(n_components=word_n)
#     word_features_train = word_pca.fit(words_tfidf_features)
#     print(f"N=={word_n} Word Explained Variance: ", sum(word_pca.explained_variance_ratio_))