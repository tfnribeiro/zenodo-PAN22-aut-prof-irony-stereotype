import numpy as np
from zmq import THREAD_SCHED_POLICY_DFLT
from read_files import *
import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split

#profanity list from:
#https://github.com/zacanger/profane-words

#total profanity
#OR number of tweets with profanity 
#try both!

prof_file = open('profanity_list.txt', 'r', encoding= 'utf-8')
prof_list = {word.rstrip() for word in prof_file.readlines()}
prof_file.close()

#preprocessing 
X = np.char.lower(X) #lowercase
X = np.char.strip(X, string.punctuation) #stripping

# user_counts = []
# for user in X:
#     freq = FreqDist()
#     for tweet in user:
#         for word in word_tokenize(tweet):
#             freq[word] += 1
#     profanity = prof_list.intersection(set(freq.keys()))
#     count = 0
#     for word in profanity:
#         count += freq[word]
#     user_counts.append(count)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Total sentences == 200
best_acc = -1
best_parameters = (-1)
profanity_threshhold = np.arange(5,100,5)
for threshold in profanity_threshhold:
    acc = -1
    correct = 0
    for i in range(len(X_train)):
        test_user = X_train[i]
        count_prof = 0
        for tweet in test_user:
            profanity = prof_list.intersection(set(tweet.split()))
            
            if len(profanity) > 0: 
                #print(profanity, len(profanity))
                count_prof += 1
                
        label = "NI"
        if count_prof > threshold:
            label = "I"

        if label == y_train[i]:
            correct += 1

    acc = correct/len(y_train)
    print("Parameters: ", (threshold), "Acc: ", correct/len(y_train))
    if acc > best_acc:
        print("In if", best_acc)
        best_parameters = threshold
        best_acc = acc
print("Current best params: ", best_parameters)
print("Best Parameters: ", best_parameters, "Acc: ", best_acc)

"""
Parameters:  5 Acc:  0.43197278911564624
In if -1
Parameters:  10 Acc:  0.41496598639455784
Parameters:  15 Acc:  0.4013605442176871
Parameters:  20 Acc:  0.35034013605442177
Parameters:  25 Acc:  0.3197278911564626
Parameters:  30 Acc:  0.3707482993197279
Parameters:  35 Acc:  0.4489795918367347
In if 0.43197278911564624
Parameters:  40 Acc:  0.47619047619047616
In if 0.4489795918367347
Parameters:  45 Acc:  0.47959183673469385
In if 0.47619047619047616
Parameters:  50 Acc:  0.4965986394557823
In if 0.47959183673469385
Parameters:  55 Acc:  0.5
In if 0.4965986394557823
Parameters:  60 Acc:  0.5
Parameters:  65 Acc:  0.5068027210884354
In if 0.5
Parameters:  70 Acc:  0.5102040816326531
In if 0.5068027210884354
Parameters:  75 Acc:  0.5136054421768708
In if 0.5102040816326531
Parameters:  80 Acc:  0.5204081632653061
In if 0.5136054421768708
Parameters:  85 Acc:  0.5204081632653061
Parameters:  90 Acc:  0.5204081632653061
Parameters:  95 Acc:  0.5170068027210885
Current best params:  80
Best Parameters:  80 Acc:  0.5204081632653061
"""