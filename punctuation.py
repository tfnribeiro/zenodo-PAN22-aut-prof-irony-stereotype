import re
import numpy as np

def count_punctuation(tweetlist):
    '''
    This function counts the repeated and normal punctuation
    :param: tweetlist list of tweets
    :return: number of normal and repeated punctuation
    '''
    punctu_count = 0
    weird_count = 0

    for tweet in tweetlist:
        #count repeated punctuation and ".*"
        weird = re.findall('(\.\.+|\?\?+|\!\!+|".*")', tweet)
        weird_count += len(weird)
        re.sub('(\.\.+|\?\?+|\!\!+)', '', tweet)

        #count regular punctuation
        all = re.findall('(\.|\?|\!)', tweet)
        punctu_count += len(all)
        
    return punctu_count, weird_count

def count_punctuation_all(all_user):
    '''
    count the punctuation for all user
    :param: list of lists of tweets
    :return: punctuation counts for all user
    '''
    punct = []
    for user in all_user:
        all, weird = count_punctuation(user)
        punct.append([all, weird])
    
    return np.array(punct)