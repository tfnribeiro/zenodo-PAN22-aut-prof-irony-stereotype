import numpy as np
from utils import *
from nltk.tag import pos_tag

"""
https://www.nltk.org/book/ch05.html

Universal tags per NLTK: 
Tag 	Meaning 	English Examples
--------------------------------------------------------
ADJ 	adjective 	new, good, high, special, big, local
ADP 	adposition 	on, of, at, with, by, into, under
ADV 	adverb 	    really, already, still, early, now
CONJ 	conjunction and, or, but, if, while, although
DET 	determiner 	the, a, some, most, every, no, which
NOUN 	noun 	    year, home, costs, time, Africa
NUM 	numeral 	twenty-four, fourth, 1991, 14:24
PRT 	particle 	at, on, out, over per, that, up, with
PRON 	pronoun 	he, their, her, its, my, I, us
VERB 	verb 	    is, say, told, given, playing, would
. 	punctuation marks 	. , ; !
X 	other 	ersatz, esprit, dunno, gr8, univeristy

"""
def pos_counts(author_tweet_list):
    pos_dict = {
        "ADJ":0,
        "ADP":0,
        "ADV":0,
        "CONJ":0,
        "DET":0,
        "NOUN":0,
        "NUM":0,
        "PRT":0,
        "PRON":0,
        "VERB":0,
        ".":0,
        "X":0
    }
    for tweet in author_tweet_list:
        for word, tag in pos_tag(tokenize_tweet(tweet, True), tagset='universal'):
            pos_dict[tag] += 1
    return np.array(list(pos_dict.values()))/len(author_tweet_list)
