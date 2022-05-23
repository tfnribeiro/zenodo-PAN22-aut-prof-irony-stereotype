from utils import *
from classifier_methods import *

X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))
X_test, USERCODE_TEST_X, lang_test = load_dataset(os.path.join(os.getcwd(),"data","test","en"), is_test=True)

classifier, *settings = train_model(X,y)
X_test_features = get_features_test(X_test, *settings)

pred = classifier.predict(X_test_features)
generate_predictions_output(pred, USERCODE_TEST_X, lang_test, "test_labels")