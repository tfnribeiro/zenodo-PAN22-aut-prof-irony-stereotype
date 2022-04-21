from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from read_files import *

sid = SentimentIntensityAnalyzer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

def get_sent_polarity(user_tweet_list, ironic_setence_t, polarity_t):
    count_I = 0
    polarity = 0 
    for tweet in user_tweet_list:
        sent_dict = sid.polarity_scores(tweet)
        polarity = np.abs(sent_dict['pos']-sent_dict['neg'])
        if polarity < polarity_t and sent_dict['pos'] > 0 and sent_dict['neg'] > 0:
            count_I += 1
    if count_I > ironic_setence_t:
        return "I"
    return "NI"

# Total sentences == 200
best_acc = -1
best_parameters = (-1,-1)
sentence_n_list = np.arange(10,91,10)
polarity_threshold_list = np.arange(0.1, 0.4, 0.05)
for sentence_n in sentence_n_list:
    for polarity_t in polarity_threshold_list:
        correct = 0
        for i in range(len(X_train)):
            test_user = X_train[i]
            label = get_sent_polarity(test_user, sentence_n, polarity_t)
            if label == y_train[i]:
                correct += 1
        acc = correct/len(y_train)
        print("Parameters: ", (sentence_n, polarity_t), "Acc: ", correct/len(y_train))
        if acc > best_acc:
            best_parameters = (sentence_n, polarity_t)
            best_acc = acc
    print("Current best params: ", best_parameters)

print("Best Parameters: ", best_parameters, "Acc: ", best_acc)

correct = 0
for i in range(len(X_test)):
    test_user = X_test[i]
    label = get_sent_polarity(test_user, sentence_n, polarity_t)
    if label == y_test[i]:
        correct += 1
acc = correct/len(y_test)
print("Test Results: ", correct, len(y_test), "Acc: ", correct/len(y_test))

"""
Best Parameters:  (40, 0.01) Acc:  0.6666666666666666
Test Results:
Correct:86 Total:126 | Acc:  0.6825396825396826

Current best params:  (35, 0.001)
Best Parameters:  (35, 0.001) Acc:  0.6996336996336996
93 147 Acc:  0.6326530612244898

Current best params:  (42.5, 0.0121)
Best Parameters:  (42.5, 0.0121) Acc:  0.6703296703296703
101 147 Acc:  0.6870748299319728

"""