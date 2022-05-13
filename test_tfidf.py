from tfidf import *

corpus = np.array([["this is very fun.", "I see see the fun in this.", "This"]])
print(corpus.shape)
tfidf_class = tfidf(corpus)

print("No Flags Test: ")
print("IDF VECTOR: ", tfidf_class.term_idf)
print(tfidf_class.tf_idf(["that"]).max(), 0)
print(tfidf_class.tf_idf(["fun"]).max(), np.log2(3/2) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["this"]).max(), np.log2(3/2) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["seen"]).max(), 0)
print(tfidf_class.tf_idf(["this is this"]).max(), np.log2(3/1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["see see I told you."]).max(), np.log2(3/1) * (1 + np.log2(2)))

tfidf_class = tfidf(corpus, authors_document=True)

print("Author Documents Test: ")
print("IDF VECTOR: ", tfidf_class.term_idf)
print(tfidf_class.tf_idf(["that"]).max(), 0)
print(tfidf_class.tf_idf(["fun"]).max(), np.log2(1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["this"]).max(), np.log2(1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["seen"]).max(), 0)
print(tfidf_class.tf_idf(["this is this"]).max(), np.log2(1) * (1 + np.log2(2)))
print(tfidf_class.tf_idf(["see see I told you."]).max(), np.log2(1) * (1 + np.log2(2)))

author_2_corpus = np.vstack((corpus, np.array([["this","",""]])))
print(author_2_corpus.shape)

tfidf_class = tfidf(author_2_corpus, authors_document=True)

print("Author Documents Test v2: ")
print("IDF VECTOR: ", tfidf_class.term_idf)
print(tfidf_class.tf_idf(["that"]).max(), 0)
print(tfidf_class.tf_idf(["fun"]).max(), np.log2(2/1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["this"]).max(), np.log2(1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["seen"]).max(), 0)
print(tfidf_class.tf_idf(["this is this"]).max(), np.log2(2/1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["see see I told you."]).max(), np.log2(2/1) * (1 + np.log2(2)))

tfidf_class = tfidf(corpus, lowercase=True)

print("Lowecase Test: ")
print("IDF VECTOR: ", tfidf_class.term_idf)
print(tfidf_class.tf_idf(["that"]).max(), 0)
print(tfidf_class.tf_idf(["fun"]).max(), np.log2(3/2) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["this"]).max(), np.log2(3/3) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["seen"]).max(), 0)
print(tfidf_class.tf_idf(["this is this"]).max(), np.log2(3/1) * (1 + np.log2(1)))
print(tfidf_class.tf_idf(["see see I told you."]).max(), np.log2(3/1) * (1 + np.log2(2)))


