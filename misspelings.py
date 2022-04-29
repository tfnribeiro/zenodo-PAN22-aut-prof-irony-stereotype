from spellchecker import SpellChecker
from utils import tweet_to_wordlist




def misspelled(tweetlist):
    sp = SpellChecker()
    misspells = 0
    word_count = 0
    for tweet in tweetlist:
        words = tweet_to_wordlist(tweet)
        miss = sp.unknown(words)
        misspells += len(miss)
        word_count += len(words)
    
    return misspells/word_count


def all_misspelled(all_user):
    missrates = []
    for user in all_user:
        missrate = misspelled(user)
        missrates.append(missrate)
    
    return missrates