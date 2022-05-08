from spellchecker import SpellChecker
from utils import tweet_to_wordlist
import numpy as np

def misspelled(tweetlist):
    sp = SpellChecker()
    misspells = 0
    word_count = 0
    for tweet in tweetlist:
        words = tweet_to_wordlist(tweet)
        miss = sp.unknown(words)
        misspells += len(miss)
        word_count += len(words)
    if word_count == 0:
        return misspells/1
    return misspells/word_count

def all_misspelled(all_user):
    missrates = np.zeros(len(all_user))
    for i, user in enumerate(all_user):
        missrate = misspelled(user)
        missrates[i] = missrate
    return missrates