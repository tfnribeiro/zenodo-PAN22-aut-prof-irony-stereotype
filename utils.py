from nltk.tokenize import TweetTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import os
import numpy as np
import glob
import re
from xml.etree import ElementTree as ET
from io import StringIO
from timeit import default_timer as timer
from datetime import datetime, timedelta

class stopwatch:
    def __init__(self):
        self.start = timer()
        self.last_split = None
    def current_time(self):
        time = datetime.now()
        return time.strftime("%d/%m/%Y %H:%M:%S")
    def split(self, as_string=False):
        if self.last_split is None:
            self.last_split = timer()
            current_split = timer() - self.start
        else: 
            current_split = timer() - self.last_split
            self.last_split = timer()
        if as_string:
            return str(timedelta(seconds=current_split))
        return current_split
    def elapsed(self, as_string=False):
        elapsed_time = timer() - self.start
        if as_string:
            return str(timedelta(seconds=elapsed_time))
        return elapsed_time

def tokenize_tweet(tweet_to_tokenize, as_list=False):
    tknzr = TweetTokenizer()
    tweet_to_tokenize = tweet_to_tokenize.replace("#USER#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#USER#".lower(),"")
    tweet_to_tokenize = tweet_to_tokenize.replace("#HASHTAG#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#HASHTAG#".lower(),"")
    tweet_to_tokenize = tweet_to_tokenize.replace("#URL#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#URL#".lower(),"")
    if as_list:
        return tknzr.tokenize(tweet_to_tokenize)
    return " ".join(tknzr.tokenize(tweet_to_tokenize))

def get_corpus(list_of_authors):
    """
        Returns the corpus as a single string.
    """
    buf = StringIO()
    for author in list_of_authors:
        for tweet in author:
            buf.write(tweet + " ")
    return buf.getvalue()

def tweet_to_wordlist(tweet):
    tweet = tweet.encode("ascii", "ignore").decode()
    tweet = tweet.replace("#USER#","")
    tweet = tweet.replace("#HASHTAG#","")
    tweet = tweet.replace("#URL#","")
    tweet = re.sub('(???|???)', '"', tweet)
    tweet = re.sub("(???|???)", "'", tweet)
    tweet = re.sub("( '|' |???|&amp;)", "", tweet)
    tweet = re.sub('(\.|\?|\!|"|,|\(|\)|:)', "", tweet)
    
    return tweet.split()

def copy_dictionary(dic_to_copy):
    return {k:v for k, v in dic_to_copy.items()}

def filter_dictionary(dictionary, filter_list=[]):
    for keys_del in filter_list:
        del dictionary[keys_del]
    return dictionary

def load_dataset(DATASET_DIR, is_test=False):
    def get_representation_tweets(F):
        parsedtree = ET.parse(F)
        documents = parsedtree.iter("document")

        texts = []
        for doc in documents:
            texts.append(doc.text)

        return texts
    if not is_test:
        GT = os.path.join(DATASET_DIR, "truth.txt")
        true_values = {}
        with open(GT) as f:
            for line in f:
                linev = line.strip().split(":::")
                true_values[linev[0]] = linev[1]
    
    lang = DATASET_DIR.split(os.sep)[-1]
    USERCODE_X = []
    X = []

    if not is_test:
        y = []

    for FILE in glob.glob(os.path.join(DATASET_DIR,"*.xml")):
        #The split command below gets just the file name,
        #without the whole address. The last slicing part [:-4]
        #removes .xml from the name, so that to get the user code
        USERCODE = os.path.split(FILE)[-1][:-4]
        #This function should return a vectorial representation of a user
        repr = get_representation_tweets(FILE)
        USERCODE_X.append(USERCODE)
        #We append the representation of the user to the X variable
        #and the class to the y vector
        try:
            X.append(repr)
            if not is_test:
                y.append(true_values[USERCODE])
        except:
            print("Failed to find: ", USERCODE)

    X = np.array(X)
    if not is_test:
        y = np.array(y)
    USERCODE_X = np.array(USERCODE_X)
    print("Load XML files complete, number of tweet profiles: ", len(X))
    if is_test:
        return X, USERCODE_X, lang
    else:
        return X, y, USERCODE_X, lang

def generate_predictions_output(predictions, usercode_x, lang="en", output_dir="OUTPUT"):
    """ 
        Format desired:<author id="author-id" "lang="en" type="NI|I"/>
        One file per author
    """
    curr_path = os.getcwd()
    output_dir_path = os.path.join(curr_path,output_dir)
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    
    for user_i, usercode in tqdm(enumerate(usercode_x)):
        user_file = os.path.join(output_dir_path,usercode+".xml")
        with open(user_file, "w") as f:
            line_out = f"""<author id="{usercode}" lang="{lang}" type="{predictions[user_i]}"/>"""
            f.write(line_out)
        print(f"File written to: {user_file}")
    print("Creating output files done in: ", output_dir_path)
    
def get_top_pca_values(pca_class, tf_idf_class, top_n=20):
    ind = np.argpartition(pca_class.components_.mean(axis=0), -top_n)[-top_n:]
    return [tf_idf_class.index_to_term[i] for i in ind]

def get_top_idf_values(author_list, y_train, tf_idf_class, top_n=20):
    n_authors, _ = author_list.shape
    tf_idf_matrix = []
    for author in tqdm(author_list):
        tf_idf_matrix.append(tf_idf_class.tf_idf(author))
    pipe_log = LogisticRegression() #make_pipeline(StandardScaler(), LogisticRegression())
    print(pipe_log)
    pipe_log.fit(tf_idf_matrix, y_train)
    #log_reg_class = pipe_log.named_steps['logisticregression']
    log_weights = pipe_log.coef_
    print(log_weights.shape)
    log_weights = log_weights[0]
    print(log_weights.shape)
    ind = np.argpartition(log_weights, -top_n)[-top_n:]
    print(ind)
    return [tf_idf_class.index_to_term[i] for i in ind]



