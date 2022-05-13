from classifier_methods import *
from utils import * 

X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))
#
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#
##x_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X)
#
##generate_predictions_output(predictions, USERCODE_X, lang)
#
#cross_val_acc_dict_9, cross_val_f1_dict_9= cross_validate(X, y, 9)
#cross_val_acc_dict_7, cross_val_f1_dict_7 = cross_validate(X, y, 7)
#cross_val_acc_dict_5, cross_val_f1_dict_5 = cross_validate(X, y, 5)
cross_val_acc_dict_3, cross_val_f1_dict_3 = cross_validate(X, y, 3)

#print_dictionaries_cross_validate(cross_val_acc_dict_9, cross_val_f1_dict_9, 9)
#print_dictionaries_cross_validate(cross_val_acc_dict_7, cross_val_f1_dict_7, 7)
#print_dictionaries_cross_validate(cross_val_acc_dict_5, cross_val_f1_dict_5, 5)
print_dictionaries_cross_validate(cross_val_acc_dict_3, cross_val_f1_dict_3, 3)