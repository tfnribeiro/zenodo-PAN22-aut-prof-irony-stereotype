from classifier_methods import *
from utils import * 

X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80)
#
##x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X)
#
##generate_predictions_output(predictions, USERCODE_X, lang)
# Emoji: 65, Profanity: 95,  Word: 255, 90% Variance (All X, this might not hold for different folds)
# predictions, *rest, classifier = generate_features_train_predict(X_train, y_train, X_test)

acc_dict, f1_dict, best_e, best_p, best_w = cross_validate_tune_params(X, y, 5, emoji_pca_dim=np.arange(5,10), profanity_pca_dim=np.arange(10,15), word_pca_dim=np.arange(20,110,20))


# emoji_tfidf = fit_emoji_embeds_tfidf(
#     X, authors_document=False)
# emoji_tfidf_features = get_features(
#     X, emoji_tfidf.tf_idf, "Emoji TF_IDF")
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