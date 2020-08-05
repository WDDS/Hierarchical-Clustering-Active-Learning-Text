import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
#import spacy
import re
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import logging

N_CLUSTERS = 256
OUTPUT_DIR = "/space/Work/adrian/output/"#"/home/adrian/workspace/Hierarchical-Clustering-Active-Learning-Text/outputs/"
VECTORS_FILE = "/space/tmp/pubmed2018_w2v_200D.bin" #"/home/adrian/PhD/Data/Word2Vec/BioASQvectors2018/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin"
DATA_DIR = "/space/Work/adrian/baseline_diabetes_unique_maxNperClass5000.parquet" #"/home/adrian/PhD/Data/Pubmed/baseline_diabetes_unique.parquet"
LOGFILE = "logfile.txt"


logging.basicConfig(filename=OUTPUT_DIR+LOGFILE,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

#define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
#formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)

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



logging.info("Load model..")
model = KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
logging.info("Load data..")
df = pd.read_parquet(DATA_DIR)


index2word_set = set(model.wv.index2word)

# clean for BioASQ
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

df_vec = np.stack(
                    (df.title + " " + df.abstract).map(lambda abstract: avg_feature_vector(abstract, model, index2word_set)).values
                    , axis = 0)

logging.info("Vector shape: {}".format(df_vec.shape))

#df_vec = cosine_similarity(df_vec)
#print("Similarity matrix shape: {}".format(df_vec.shape))

HC = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity="cosine", linkage="average")
logging.info("Fit..")
HC.fit(df_vec)

logging.info("Number of leaves: {}".format(HC.n_leaves_))
logging.info("Number of clusters: {}".format(HC.n_clusters_))

# https://stackoverflow.com/questions/27386641/how-to-traverse-a-tree-from-sklearn-agglomerativeclustering
ii = itertools.count(df_vec.shape[0])
tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in HC.children_]

df["class_predict"] = HC.labels_
tree_df = pd.DataFrame(tree, columns=["node_id", "left", "right"])

logging.info("Save results to file..")
df.to_parquet(OUTPUT_DIR+"/diabetes_abstracts_HC_output.parquet")
tree_df.to_parquet(OUTPUT_DIR+"/diabetes_abstracts_tree_output.parquet")

logging.info("All good :)")
