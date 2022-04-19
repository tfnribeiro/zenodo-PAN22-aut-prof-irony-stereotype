import os
import numpy as np
from os import path
import glob

#Library to open XML files
from xml.etree import ElementTree as ET

#Setting some variables
#Get home directory
#HOME = os.environ['HOME']
DATA_HOME = path.join("data", "en")
#Set the directory where we saved the corpus of fake news spreaders
filename = "6efbf0c5385bde90583d02f14750f7d9.xml"
filename = path.join(DATA_HOME, filename)

#Set the language
LANG  = "en/"

#simple test
def get_representation_tweets(F):

    parsedtree = ET.parse(F)
    documents = parsedtree.iter("document")

    texts = []
    for doc in documents:
        texts.append(doc.text)

    #HERE YOU SHOULD PROCESS THOSE TWEETS AND RETURN THE REPRESENTATION
    return texts

GT    = path.join(DATA_HOME, "truth.txt")
true_values = {}
f=open(GT)
for line in f:
    linev = line.strip().split(":::")
    true_values[linev[0]] = linev[1]
f.close()

X = []
y = []

print(DATA_HOME)
for FILE in glob.glob(DATA_HOME+"\*.xml"):
    #The split command below gets just the file name,
    #without the whole address. The last slicing part [:-4]
    #removes .xml from the name, so that to get the user code
    USERCODE = FILE.split("\\")[-1][:-4]

    print(FILE)
    #This function should return a vectorial representation of a user
    repr = get_representation_tweets(FILE)

    #We append the representation of the user to the X variable
    #and the class to the y vector
    try:
        X.append(repr)
        y.append(true_values[USERCODE])
    except:
        print("Failed to find: ", USERCODE)

X = np.array(X)
y = np.array(y)

