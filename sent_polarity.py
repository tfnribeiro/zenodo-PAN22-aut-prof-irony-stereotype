from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from read_files import *

sid = SentimentIntensityAnalyzer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Total sentences == 200
best_acc = -1
best_parameters = (-1,-1)
sentence_n_list = np.arange(25,100,15)
polarity_threshold_list = np.arange(0.01, 0.11, 0.01)
for sentence_n in sentence_n_list:
    for polarity_t in polarity_threshold_list:
        acc = -1
        correct = 0
        for i in range(len(X_train)):
            test_user = X_train[i]
            count_I = 0
            polarity = 0 
            for tweet in test_user:
                sent_dict = sid.polarity_scores(tweet)
                polarity = np.abs(sent_dict['pos']-sent_dict['neg'])
                if polarity < polarity_t:
                    count_I += 1
            label = "NI"
            if count_I > sentence_n:
                label = "I"
            #print(y[i], "Predicted: ", label, "| Counts:",count_I)
            if label == y_train[i]:
                correct += 1
        acc = correct/len(y_train)
        print("Parameters: ", (sentence_n, polarity_t), "Acc: ", correct/len(y_train))
        if acc > best_acc:
            print("In if", best_acc)
            best_parameters = (sentence_n, polarity_t)
            best_acc = acc
    print("Current best params: ", best_parameters)

print("Best Parameters: ", best_parameters, "Acc: ", best_acc)

correct = 0
for i in range(len(X_test)):
    test_user = X_test[i]
    count_I = 0
    polarity = 0 
    for tweet in test_user:
        sent_dict = sid.polarity_scores(tweet)
        polarity = np.abs(sent_dict['pos']-sent_dict['neg'])
        if polarity < best_parameters[1]:
            count_I += 1
    label = "NI"
    if count_I > best_parameters[0]:
        label = "I"
    #print(y[i], "Predicted: ", label, "| Counts:",count_I)
    if label == y_test[i]:
        correct += 1
acc = correct/len(y_test)
print(correct, len(y_test), "Acc: ", correct/len(y_test))

"""
Best Parameters:  (40, 0.01) Acc:  0.6666666666666666
Test Results:
Correct:86 Total:126 | Acc:  0.6825396825396826

"""