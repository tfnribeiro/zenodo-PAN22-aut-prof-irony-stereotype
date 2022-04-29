import numpy as np
from zmq import THREAD_SCHED_POLICY_DFLT
from read_files import *
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import SparsePCA #used for sparse data 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from nltk.stem.porter import *

#profanity list from:
#https://github.com/zacanger/profane-words

#total profanity
#OR number of tweets with profanity 
#try both!

prof_file = open('profanity_list.txt', 'r', encoding= 'utf-8')
prof_list = [word.rstrip() for word in prof_file.readlines()]
prof_file.close()

#preprocessing 
X = np.char.lower(X) #lowercase
X = np.char.strip(X, string.punctuation) #stripping

#count array
# count_array = np.full((X.shape[0], len(prof_list)), 0)
# stemmer = PorterStemmer()

# for n, user in enumerate(X):
#     freq = FreqDist()
#     for tweet in user:
#         for word in word_tokenize(tweet):
#             word = stemmer.stem(word)
#             freq[word] += 1
#     profanity = set(prof_list).intersection(set(freq.keys()))
#     for word in profanity:
#         count_array[n, prof_list.index(word)] = freq[word]

# # #save count array to dataframe csv
# df = pd.DataFrame(count_array, columns = prof_list)
# df.to_csv('profanity_counts.csv')


#DATA VISUALIZATION
#STEMMING? 

# #most frequent profane words for each category 
count_array = pd.read_csv('profanity_counts.csv', index_col=[0], delimiter = ',')
i_top30 = count_array.loc[(np.where(y=='I')[0])].sum(axis = 0).sort_values(ascending = False).head(30)
ni_top30 = count_array.loc[(np.where(y=='NI')[0])].sum(axis = 0).sort_values(ascending = False).head(30)

ploti = i_top30.plot(kind = 'barh', title = "Ironic Profanity")
ploti.get_figure().savefig('plot_i.png')

plotni = ni_top30.plot(kind = 'barh', title = "Non-Ironic Profanity")
plotni.get_figure().savefig('plot_ni.png')


#PCA dimensionality reduction 
# pca = SparsePCA(n_components = 10, random_state=0).fit(count_array)
# count_array_trans = pca.transform(count_array)
# np.savetxt('profanity_counts_pca.csv', count_array_trans, delimiter = ',')

#PCA plot - 2 principal components!
#pca = SparsePCA(n_components = 2, random_state=0).fit(count_array)
# count_array_trans = pca.transform(count_array)
# plt.figure()
# plt.scatter(count_array_trans[np.where(y=='I')[0],0], count_array_trans[np.where(y=='I')[0],1],color = 'red', label = 'I')
# plt.scatter(count_array_trans[np.where(y=='NI')[0],0], count_array_trans[np.where(y=='NI')[0],1],color = 'blue', label = 'NI')
# plt.xlim((-5,30))
# plt.ylim((-5,30))
# plt.savefig('pca2plot.png')


#Random Forest Classifier - 0.8 accuracy???
prof_pca = np.loadtxt('profanity_counts_pca.csv', delimiter = ',')
# X_train, X_test, y_train, y_test = train_test_split(prof_pca, y, test_size=0.3)

# rf = RandomForestClassifier(max_depth=3)
# rf.fit(X_train, y_train)
# results = rf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, results))

# kf = KFold(n_splits=5)
# accuracies = []
# rf = RandomForestClassifier(max_depth=3)
# for train, test in kf.split(prof_pca):
#     X_train, X_test, y_train, y_test = prof_pca[train], prof_pca[test], y[train], y[test]
#     rf = RandomForestClassifier(max_depth=3)
#     rf.fit(X_train, y_train)
#     results = rf.predict(X_test)
#     accuracies.append(accuracy_score(y_test, results))
# print(accuracies)
# print(np.mean(accuracies))




# Total sentences == 200
# best_acc = -1
# best_parameters = (-1)
# profanity_threshhold = np.arange(5,100,5)
# counts = []
# for threshold in profanity_threshhold:
#     acc = -1
#     correct = 0
#     for i in range(len(X_train)):
#         test_user = X_train[i]
#         count_prof = 0
#         freq = FreqDist()
#         for tweet in test_user:
#             for word in word_tokenize(tweet):
#                 freq[word] += 1
#         profanity = prof_list.intersection(set(freq.keys()))
#         for word in profanity:
#             count_prof += freq[word]
        
#         counts.append(count_prof)
#         label = "NI"
#         if count_prof > threshold:
#             label = "I"
    
#         if label == y_train[i]:
#             correct += 1

#         print(profanity, y_train[i])

#     acc = correct/len(y_train)
#     print("Parameters: ", (threshold), "Acc: ", correct/len(y_train))
#     if acc > best_acc:
#         print("In if", best_acc)
#         best_parameters = threshold
#         best_acc = acc
# print("Current best params: ", best_parameters)
# print("Best Parameters: ", best_parameters, "Acc: ", best_acc)

# counts = np.array(counts)
# y_train = np.array(y_train)

# print("Average I: ", np.mean(counts[np.where(y_train == 'I')[0]]))
# print("Average NI: ", np.mean(counts[np.where(y_train == 'NI')[0]]))


"""
Parameters:  5 Acc:  0.48639455782312924
In if -1
Parameters:  10 Acc:  0.445578231292517
Parameters:  15 Acc:  0.4217687074829932
Parameters:  20 Acc:  0.40476190476190477
Parameters:  25 Acc:  0.3673469387755102
Parameters:  30 Acc:  0.3129251700680272
Parameters:  35 Acc:  0.30612244897959184
Parameters:  40 Acc:  0.2925170068027211
Parameters:  45 Acc:  0.3231292517006803
Parameters:  50 Acc:  0.35714285714285715
Parameters:  55 Acc:  0.3877551020408163
Parameters:  60 Acc:  0.41496598639455784
Parameters:  65 Acc:  0.43197278911564624
Parameters:  70 Acc:  0.46258503401360546
Parameters:  75 Acc:  0.46598639455782315
Parameters:  80 Acc:  0.47278911564625853
Parameters:  85 Acc:  0.47619047619047616
Parameters:  90 Acc:  0.47278911564625853
Parameters:  95 Acc:  0.47959183673469385"""
