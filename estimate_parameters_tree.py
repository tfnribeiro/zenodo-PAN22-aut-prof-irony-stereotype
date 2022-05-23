from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
from utils import *
from classifier_methods import *

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# np.random.seed(1)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions != test_labels)
    accuracy = (predictions==test_labels).sum()/len(test_labels)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))

X_train_split, X_test_split, y_train, y_test = train_test_split(X,y,test_size=0.3) 

X_train, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf = get_features_train(X_train_split)

X_test = get_features_test(X_test_split, emoji_pca, profanity_pca, word_pca, emoji_tfidf, profanity_tfidf, words_tfidf, label="Generating Test Features", supress_prints_flag=False)

clf_rfc = RandomForestClassifier()

clf_rfc.fit(X_train, y_train)
base_accuracy = evaluate(clf_rfc, X_test, y_test)

clf_rfc = RandomForestClassifier()

param_grid = {
    'bootstrap': [True],
    'max_depth': np.arange(10,101,10),
    'max_features': np.arange(1,10,2),
    'min_samples_leaf': np.arange(1,10,2),
    'min_samples_split': np.arange(1,10,2),
    'n_estimators': np.arange(100,1001, 200)
}

grid_search = GridSearchCV(estimator = clf_rfc, param_grid = param_grid, cv = 5, n_jobs = 2, verbose = 2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

# {'bootstrap': True, 'max_depth': 50, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 300}