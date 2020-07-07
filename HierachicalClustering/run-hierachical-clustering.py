import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
#import spacy
import re
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import itertools

N_CLUSTERS = 256
OUTPUT_DIR = "/home/adrian/workspace/Hierarchical-Clustering-Active-Learning-Text/outputs/"
VECTORS_FILE = "/home/adrian/PhD/Data/Word2Vec/BioASQvectors2018/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin"
DATA_DIR = "/home/adrian/PhD/Data/Pubmed/baseline_diabetes_unique.parquet"

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def avg_feature_vector(sentence, model, index2word_set, num_features=200):
    try:
        words = bioclean(sentence)
    except:
        print("bioclean did not work for: {}".format(sentence))
        print(type(sentence))
        print(math.isnan(sentence))
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
        #else:
        #    if hasNumbers(word):
        #        print("word not in vocabulary: {}".format(word))
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec



print("Load model..")
model = KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
print("Load data..")
df = pd.read_parquet(DATA_DIR)

index2word_set = set(model.wv.index2word)

# clean for BioASQ
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

df_vec = np.stack(
                    (df.title + " " + df.abstract).map(lambda abstract: avg_feature_vector(abstract, model, index2word_set)).values
                    , axis = 0)

print("Vector shape: {}".format(df_vec.shape))

df_vec = cosine_similarity(df_vec)
print("Similarity matrix shape: {}".format(df_vec.shape))

HC = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity="cosine", linkage="average")
print("Fit..")
HC.fit(df_vec)

print("Number of leaves: {}".format(HC.n_leaves_))
print("Number of clusters: {}".format(HC.n_clusters_))

# https://stackoverflow.com/questions/27386641/how-to-traverse-a-tree-from-sklearn-agglomerativeclustering
ii = itertools.count(df_vec.shape[0])
tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in HC.children_]

df["class_predict"] = HC.labels_
tree_df = pd.DataFrame(tree, columns=["node_id", "left", "right"])

print("Save results to file..")
df.to_parquet(OUTPUT_DIR+"/diabetes_abstracts_HC_output.parquet")
tree_df.to_parquet(OUTPUT_DIR+"/diabetes_abstracts_tree_output.parquet")

print("All good :)")
