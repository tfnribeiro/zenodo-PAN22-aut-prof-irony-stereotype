import re
import numpy as np

def seperated_punctuation(tweetlist):
    '''
    This function counts the repeated and normal punctuation seperatly
    :param: tweetlist list of tweets
    :return: number of all punctuation groups array(7,)
    '''
    count_vector = np.zeros(7)
    for tweet in tweetlist:
        
        multex = re.findall('\!\!+', tweet)
        multqu = re.findall('\?\?+', tweet)
        multpe = re.findall('\.\.+', tweet)
        quote = re.findall('( "| “)', tweet)
        
        #replace all special punctuations
        tweet = re.sub('(\.\.+|\?\?+|\!\!+)', '', tweet)
        
        ex = re.findall('\!', tweet)
        qu = re.findall('\?', tweet)
        pe = re.findall('\.', tweet)
        count_vector = count_vector + np.array([len(multex), len(multqu), len(multpe), len(quote), len(ex), len(qu), len(pe)])
        
    #return array with the puctuation counts
    return count_vector / tweetlist.shape[0]

def count_punctuation(tweetlist):
    '''
    This function counts the repeated and normal punctuation
    :param: tweetlist list of tweets
    :return: number of normal and repeated punctuation
    '''
    punctu_count = 0
    weird_count = 0

    for tweet in tweetlist:
        #count repeated punctuation and ""
        weird = re.findall('(\.\.+|\?\?+|\!\!+| "| “)', tweet)
        weird_count += len(weird)
        tweet = re.sub('(\.\.+|\?\?+|\!\!+)', '', tweet)
        #count other punctuation
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