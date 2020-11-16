import pandas as pd
import numpy as np
import time
from collections import Counter
import re
import math
from gensim.models import KeyedVectors
from pprint import pprint
import pyspark
from pyspark.sql.functions import isnan, when, count, col, udf
from pyspark.sql.types import *
import numpy as np
import scipy.stats as st

## CONFIGURATION ###
MAX_LEAFS = 32 # only build tree up to 32 leafs
modelDir = "/home/adrian/PhD/Epiconcept/ds-demo-feedback-explorer/src/main/html/data/models/pubmed/Clustering-diabetes-abstracts_AL_clusteredLevelUncertainitySamplingStrategy"
dataDir = "/home/adrian/workspace/ActiveLearning/Active-Learning-for-Neural-Networks/data/diabetes/2000_random_atLeast50AbstractsperMesh/2000random_1000train_noOversampling.parquet"



#######

class MeshCode:
    def __init__(self, ID, name, treeNumber, child_mesh_code=[]):
        self.id = ID
        self.name = name
        self.treeNumber = treeNumber
        self.children = child_mesh_code

    def __repr__(self):
        return "id: {}, name: {}".format(self.id, self.name)


MESH_HIERARCHY = MeshCode(
    "D003920", "Diabetes Mellitus", "C19.246" , [
        MeshCode("D048909", "Diabetes Complications", "C19.246.099", [
               MeshCode("D003925", "Diabetic Angiopathies", "C19.246.099.500", [
                       #MeshCode("D017719", "Diabetic Foot", "C19.246.099.500.191")  # to prevent double foot
                      MeshCode("D003930", "Diabetic Retinopathy", "C19.246.099.500.382")
               ])
              , MeshCode("D058065", "Diabetic Cardiomyopathies", "C19.246.099.625")
              , MeshCode("D003926", "Diabetic Coma", "C19.246.099.750", [
                       MeshCode("D006944", "Hyperglycemic Hyperosmolar Nonketotic Coma", "C19.246.099.750.490", [])
               ])
              , MeshCode("D016883", "Diabetic Ketoacidosis", "C19.246.099.812")
              , MeshCode("D003928", "Diabetic Nephropathies", "C19.246.099.875")
              , MeshCode("D003929", "Diabetic Neuropathies", "C19.246.099.937", [
                       MeshCode("D017719", "Diabetic Foot", "C19.246.099.937.250")
               ])
              , MeshCode("D005320", "Fetal Macrosomia", "C19.246.099.968")
        ])
       , MeshCode("D016640", "Diabetes, Gestational", "C19.246.200")
       , MeshCode("D003921", "Diabetes Mellitus, Experimental", "C19.246.240")
       , MeshCode("D003922", "Diabetes Mellitus, Type 1", "C19.246.267", [
                MeshCode("D014929", "Wolfram Syndrome", "C19.246.267.960")
        ])
       , MeshCode("D003924", "Diabetes Mellitus, Type 2", "C19.246.300", [
                MeshCode("D003923", "Diabetes Mellitus, Lipoatrophic", "C19.246.300.500")
        ])
       , MeshCode("D056731", "Donohue Syndrome", "C19.246.537")
       , MeshCode("D000071698", "Latent Autoimmune Diabetes in Adults", "C19.246.656")
       , MeshCode("D011236", "Prediabetic State", "C19.246.774")
    ]
)


class Tree(object):

    def __init__(self, tree_hierarchy, clusters_predict=[], mode="sklearn", sentences_all_classes=None, true_classes_all=None):
        """
        @param mode : Two possible values
            - "FBE" : Tree object for Feedback Explorer output
            - "sklearn" : Tree object for scikit learn output

        @param sentences_all_classes : List of all possible classes occuring in the sentences file (only for mode FBE)
        @param true_labels_all : All occuring true labels (mesh codes) of the documents/abstracts
        """
        self.tree = None
        if mode in ["sklearn", "FBE"]:
            self.mode = mode
        else:
            raise ValueError("Provided mode '{}' is not supported".format(mode))
        self.tree_hierarchy = tree_hierarchy # pandas dataframe with tree structure coming from hierarchical clustering
        self.n_nodes = 0 # updated by self.count_nodes()
        self.n_leafs = 0 # updated by self.count_leafs()
        self.temp_n_leafs = 1 # In mode 'FBE' helps to construct the tree with the right number of nodes
        self.clusters_predict = clusters_predict # predicted cluster for each document
        self.unique_cluster_predict = list(set(clusters_predict)) # list of all classes to calculate performance metrices
        self.leaf_nodes = [] # list of all leaf nodes
        self.sentences_all_classes = sentences_all_classes # List of all classes occuring in sentences (phrases.parquet)
        self.true_classes_documents = true_classes_all.values.tolist() # list of true labels (mesh codes) in the abstracts
        self.true_classes_documents_unique = list(set(true_classes_all)) # all possible occuring true labels (mesh codes) in the abstracts
        self.precision_all_nodes = [] # macro
        self.precision_all_nodes_weighted = []
        self.precision_all_nodes_weights = 0
        self.precision_macro = None
        self.precision_micro = None
        self.recall_all_classes = []
        self.recall_all_classes_weighted = []
        self.recall_macro = None
        self.recall_micro = None
        self.F1_macro = None
        self.F1_micro = None
        self.maxDepth = 0
        self.temp_max_occ_class_in_cluster = 0
        self.temp_max_doc_perClass_inCluster = 0
        self.temp_mesh_and_its_childs = [] # list of a given mesh code and its children mesh codes


    def _build_tree(self, node, current_depth=None):
        if self.mode == "sklearn":
            if node.node_id in self.tree_hierarchy["node_id"].values: # if node not leaf
                treeChildren = self.tree_hierarchy[self.tree_hierarchy["node_id"] == node.node_id]
                node.add_child(Node(Id=treeChildren["left"].values[0], depth=node.depth + 1, parent=node))
                node.add_child(Node(Id=treeChildren["right"].values[0], depth=node.depth + 1, parent=node))
                self._build_tree(node.children[0])
                self._build_tree(node.children[1])
            else:
                return node
            return node
        elif self.mode == "FBE":
            # Only create node if node is in current depth level
            if node.depth == current_depth and self.temp_n_leafs < MAX_LEAFS:
                treeChildren = self.tree_hierarchy.iloc[node.node_id].children
                #print("\t{}".format(node))
                #print("tree children:")
                #print(treeChildren)
                #print()
                # FBE tree is not a perfect binary tree, some nodes don't create children any more
                if len(treeChildren) > 0:
                    cluster_child_one = self.tree_hierarchy.iloc[treeChildren[0]].filterValue[0]
                    cluster_child_two = self.tree_hierarchy.iloc[treeChildren[1]].filterValue[0]
                    #print("\tc1: {}, c2: {}".format(cluster_child_one, cluster_child_two))
                    # Some nodes from nodes.json are empty: no sentences is going through them
                    # Only create node in tree when there is a sentence running through it
                    if cluster_child_one in self.sentences_all_classes:
                        #print("\t\tc1 in class")
                        self.temp_n_leafs -= 1 # lose one leaf because it is split into two new leafs
                        node.add_child(Node(Id=treeChildren[0], depth=node.depth + 1, parent=node, cluster_label=cluster_child_one))
                        self.temp_n_leafs += 1
                        if cluster_child_two in sentences_all_classes:
                            #print("\t\tc1 and c2 in class")
                            node.add_child(Node(Id=treeChildren[1], depth=node.depth + 1, parent=node, cluster_label=cluster_child_two))
                            self.temp_n_leafs += 1
                    elif cluster_child_two in sentences_all_classes:
                        #print("\t\tc2 in class")
                        self.temp_n_leafs -= 1 # lose one leaf because it is split into two new leafs
                        node.add_child(Node(Id=treeChildren[1], depth=node.depth + 1, parent=node, cluster_label=cluster_child_two))
                        self.temp_n_leafs += 1
                    #else:
                    #    print("\t\tno class for c1 and c2")
            else:
                if len(node.children) == 1 and self.temp_n_leafs < MAX_LEAFS:
                    self._build_tree(node.children[0], current_depth)
                elif len(node.children) == 2 and self.temp_n_leafs < MAX_LEAFS:
                    self._build_tree(node.children[0], current_depth)
                    self._build_tree(node.children[1], current_depth)
            return node

    def _update_leaf_to_root(self, node, abstract_id, class_predict):
        """ Updates node and all its ancestors up to the root with the abstract's id and the predicted class"""
        node.update_node(abstract_id, class_predict)
        if node.parent != None: # Root has no parent
            self._update_leaf_to_root(node.parent, abstract_id, class_predict)


    def set_build_tree(self,node):
        """ Builds the tree and sets the variable tree."""

        # tree with MAX_LEAFS leafs is constructed.
        # For sklearn add to each leaf its cluster label based on the children in the tree object from sklearn AgglomerativeClustering
        self.leaf_nodes = []
        if self.mode == "sklearn":
            tree = self._build_tree(node) # construct whole tree
            tree = self._get_cluster_labels_for_leafs(tree) # get labels for leafs
            tree = self._cut_nodes_from_leafs(tree) # cut nodes from bottom of the tree until only leafs with a unique cluster_label exist (Number leaves = MAX_LEAFS)
        elif self.mode == "FBE":
            self.temp_n_leafs = 1
            self.maxDepth = 0
            self._get_maxDepth(0, 0)
            depth = 0
            #print("maxDepth: {}".format(self.maxDepth))
            # build tree by level: create first all children for level 1, then level 2...
            # Prevents that a tree creates children just in one branch and always goes deeper in case of a max number of leavese
            while self.temp_n_leafs < MAX_LEAFS and depth <= self.maxDepth:
                print("\n\ndepth: {}, temp_n_leafs: {}, maxDepth: {}".format(depth, self.temp_n_leafs, self.maxDepth))
                tree = self._build_tree(node, depth)
                depth += 1

        assert isinstance(tree, Node)
        self.tree = tree
        print("Count nodes: {}; leafs: {}".format(self.count_nodes(), self.count_leafs()))


    def _get_maxDepth(self, i, depth):
        """ get max depth of tree"""
        if depth > self.maxDepth:
            self.maxDepth = depth
        node = self.tree_hierarchy.iloc[i]
        if len(node.children) == 1:
            self._get_maxDepth(node.children[0], depth+1)
        elif len(node.children) == 2:
            self._get_maxDepth(node.children[0], depth+1)
            self._get_maxDepth(node.children[1], depth+1)



    def _get_cluster_labels_for_leafs(self, node):
        """
            Get's the cluster labels for each leafs using the cluster labels assigned by
            the output of the sklearn agglomerative clustering algorithm.
        """
        if len(node.children) == 0: #leaf
            cluster_label = self.clusters_predict[node.node_id]
            node.set_clusterLabel(cluster_label)
        else: # no leaf
            self._get_cluster_labels_for_leafs(node.children[0])
            self._get_cluster_labels_for_leafs(node.children[1])
        return node

    def _cut_nodes_from_leafs(self, node):
        """
            self.mode == sklearn:
            Children of nodes, who are leafs and have the same cluster_label, are cut off
            and the parent node takes the cluster label of its children.
            This is done recursively until there are only leafs with unique cluster_labels
            Number of leaves = MAX_LEAFS

            self.mode == FBE:
            Towards the bottom of the tree, it may happen that a node has only child, which has only one child,
            and this child also has only one child, etc. Several nodes following of each other with only one child.
            In this case keep only the child C whose parent has two children and cut the child of C.
        """
        if self.mode == "sklearn":
            if len(node.children) > 0:
                left_child = node.children[0]
                right_child = node.children[1]
                if left_child.cluster_label is None: # left child is not leaf
                    self._cut_nodes_from_leafs(left_child)
                if right_child.cluster_label is None: # right child is not leaf
                    self._cut_nodes_from_leafs(right_child)

                # should be updated now
                left_child = node.children[0]
                right_child = node.children[1]
                if left_child.cluster_label == right_child.cluster_label and left_child.cluster_label is not None:
                    node.children = []
                    node.cluster_label = left_child.cluster_label
                    return node
        elif self.mode == "FBE":
            if len(node.children) == 1: # node has only one child
                temp = node
                while len(temp.children) == 1: # check if several nodes following of each other have only one child two
                    temp = temp.children[0]
                if len(temp.children) == 2: # if at some point a node has two children, continue to search
                    self._cut_nodes_from_leafs(temp.children[0])
                    self._cut_nodes_from_leafs(temp.children[1])
                else: # if we reached a leaf, cut the node's children
                    node.children = []
            elif len(node.children) == 2:
                self._cut_nodes_from_leafs(node.children[0])
                self._cut_nodes_from_leafs(node.children[1])
        return node



    def fitTree(self, node, data):
        """ Updates all the nodes of the tree according to the clustering from bottom to top """

        assert isinstance(node, Node)
        if len(node.children) > 0: # no leaf
            for child in node.children:
                self.fitTree(child, data)
        else: # leaf
            if self.mode == "sklearn":
                leaf_cluster_label = node.cluster_label
                abstract_hits = data[data["class_predict"] == leaf_cluster_label]
                for i, row in abstract_hits.iterrows():
                    leaf_abstract_id = row.name
                    leaf_abstract_class_true = row.mesh_ui_diab # true class
                    self._update_leaf_to_root(node, leaf_abstract_id, leaf_abstract_class_true)
            elif self.mode == "FBE": # several documents per leaf
                leaf_cluster_label = node.cluster_label
                abstract_hits = data[data["uniqueCluster"] == leaf_cluster_label]
                for i, row in abstract_hits.iterrows():
                    leaf_abstract_id = row["id"]
                    leaf_abstract_class_true = row["mesh_ui_diab"]
                    self._update_leaf_to_root(node, leaf_abstract_id, leaf_abstract_class_true)
            else:
                print("ERROR: mode should be one of ['sklearn', 'FBE']")
        return node


    def count_nodes(self, tree=None):
        self.n_nodes = 0
        def _walk_count_nodes(node):
            self.n_nodes += 1
            for child in node.children:
                _walk_count_nodes(child)

        if tree == None:
            _walk_count_nodes(self.tree)
        else:
            _walk_count_nodes(tree)
        return self.n_nodes


    def count_leafs(self, tree=None):

        def _walk_count_leafs(node):
            if node.children == []:
                self.n_leafs += 1
                self.leaf_nodes.append(node)
            else:
                for child in node.children:
                    _walk_count_leafs(child)

        self.n_leafs = 0
        self.leaf_nodes = []
        if tree == None:
            _walk_count_leafs(self.tree)
        else:
            _walk_count_leafs(tree)
        return self.n_leafs


    def get_leaf_nodes(self):
        def _walk_leaf_nodes(node):
            if node.children == []:
                self.leaf_nodes.append(node)
            else:
                for child in node.children:
                    _walk_leaf_nodes(child)

        self.leaf_nodes = []
        _walk_leaf_nodes(self.tree)
        return self.leaf_nodes

    def _walk_precision(self, node):
        node_precision = node.get_precision()
        self.precision_all_nodes.append(node_precision)
        self.precision_all_nodes_weighted.append(node_precision * node.counts)
        self.precision_all_nodes_weights += node.counts
        for child in node.children:
            self._walk_precision(child)

    def get_precision(self):
        self.precision_all_nodes = []
        self.precision_all_nodes_weighted = []
        self.precision_all_nodes_weights = 0
        self._walk_precision(self.tree)
        self.precision_macro = np.mean(self.precision_all_nodes)
        self.precision_micro = np.sum(self.precision_all_nodes_weighted) / self.precision_all_nodes_weights
        return {"prec_macro" : self.precision_macro
                , "prec_micro" : self.precision_micro}


    def get_recall(self):

        self.recall_all_classes = []
        self.recall_all_classes_weighted = []
        def _walk_recall(node, c):
            """ Get cluster with max documents of class c in which class c is the majority class """
            class_counts = Counter(node.true_classes).most_common()
            majority_classes = [c for c, occ in class_counts  if occ == class_counts[0][1]] # there can be several majority classes in a node
            #majority_class = Counter(node.true_classes).most_common()[0][0]
            occ = node.true_classes.count(c)
            #print()
            #print(node)
            #print("\t{}".format(node.true_classes))
            #print("\tmajority_classe: {}, occ({}): {}".format(majority_classes, c, occ))
            if c in majority_classes and occ > self.temp_max_occ_class_in_cluster:
                self.temp_max_occ_class_in_cluster = occ
            #    print("\t updatetemp_max_occ_class_in_cluster: {}".format(self.temp_max_occ_class_in_cluster))
            #if (occ > self.temp_max_occ_class_in_cluster
            #    and (c in majority_classes or node.children == [])
            #   ): # if we found a cluster with higher occ of documents for class c and the class c is the majority class in the cluster or leaf node
            #    self.temp_max_occ_class_in_cluster = occ
            #    print("\tupdatetemp_max_occ_class_in_cluster: {}".format(self.temp_max_occ_class_in_cluster))

            #if (occ > self.temp_max_occ_class_in_cluster and c in majority_classes):
            # self.temp_max_occ_class_in_cluster = occ
            #    print("\MAJ: tupdatetemp_max_occ_class_in_cluster: {}".format(self.temp_max_occ_class_in_cluster))
            #elif (occ > self.temp_max_occ_class_in_cluster and node.children == []):
            #    self.temp_max_occ_class_in_cluster = occ
            #    print("\tLEAF: updatetemp_max_occ_class_in_cluster: {}".format(self.temp_max_occ_class_in_cluster))

            for child in node.children:
                _walk_recall(child, c)

        weights_sum = 0
        for c in self.true_classes_documents_unique:
            N_c = self.true_classes_documents.count(c)
            #print("\nc: {}, N_c: {}".format(c, N_c))
            self.temp_max_occ_class_in_cluster = 0
#            _walk_recall(self.tree, c)
            # TODO: check if it is right!
            # # start with children; otherwise recalls for all classes will be highest in root
            _walk_recall(self.tree.children[0], c)
            _walk_recall(self.tree.children[1], c)
            recall = self.temp_max_occ_class_in_cluster / N_c
            #print("c: {}, recall: {}".format(c, recall))

            self.recall_all_classes.append(recall) #len(self.unique_cluster_predict))
            self.recall_all_classes_weighted.append(recall * N_c)
            weights_sum += N_c
        self.recall_macro = np.mean(self.recall_all_classes)
        self.recall_micro = np.sum(self.recall_all_classes_weighted) / weights_sum
        return {"recall_macro" : self.recall_macro
                ,"recall_micro" : self.recall_micro}

    def get_F1(self):
        precision = self.get_precision()
        recall = self.get_recall()

        self.F1_macro = 2*precision["prec_macro"]*recall["recall_macro"] / (precision["prec_macro"] + recall["recall_macro"])
        self.F1_micro = 2*precision["prec_micro"]*recall["recall_micro"] / (precision["prec_micro"] + recall["recall_micro"])
        return {"F1_macro":self.F1_macro
               ,"F1_micro":self.F1_micro}


    def _get_child_mesh_classes(self, meshId, currentMesh, foundMeshInHierarchy=False):
        """ For a given meshId, get all its child meshId's from meshHierarchy """
        if meshId == currentMesh.id:
            foundMeshInHierarchy = True
        if foundMeshInHierarchy:
            self.temp_mesh_and_its_child_classes.append(currentMesh)
        for mesh_child in currentMesh.children:
            self._get_child_mesh_classes(meshId, mesh_child, foundMeshInHierarchy)


    def F1_zhao(self, evaluateOnlyOnLeafs=False):
        """ F1 score like in Evaluation of Hierarchical Clustering Algorithms forDocument Datasets from Zhao & Karypis """

        def _walk_F1_zhao(node, mesh_and_child_classes, N_c, evaluateOnlyOnLeafs):
            """
                Calculates F1 Score for a given list of mesh codes and its children mesh_and_child_classes (N_c = total number of documents of class c)
                evaluateOnlyOnLeafs [True, False] : calculate F1 score only on leafs or on all nodes
            """
            #print("\t{}".format(node))
            #print("abstracts in node:")
            #print(node.true_classes)
            #for m in mesh_and_child_classes:
            #    print("\t\t mesh: {}; count mesh in node: {}".format(m, node.true_classes.count(m.id)))
            class_count = np.sum([node.true_classes.count(m.id) for m in mesh_and_child_classes])# + node.true_classes.count(childs of class c)
            prec =  class_count / node.counts
            recall = class_count / N_c #+ all documents from all children of c
            if prec > 1e-10 or recall > 1e-10: # if prec or recall == 0
                F1 = 2 * prec * recall / (prec+recall)
            else:
                F1 = 0.0
            #print("\tclass_count: {}, prec: {}, recall: {}, F1: {}".format(class_count, prec, recall, F1))
            if F1 > self.temp_max_doc_perClass_inCluster:
                self.temp_max_doc_perClass_inCluster = F1

            if not evaluateOnlyOnLeafs:
                for child in node.children:
                    _walk_F1_zhao(child, mesh_and_child_classes, N_c, evaluateOnlyOnLeafs)

        if evaluateOnlyOnLeafs:
            leafs = self.get_leaf_nodes()

        FScore_sum = 0
        for meshid in self.true_classes_documents_unique:
            self.temp_mesh_and_its_child_classes = [] # reset
            self._get_child_mesh_classes(meshid, MESH_HIERARCHY)
            mesh_and_child_classes = self.temp_mesh_and_its_child_classes
            #N_c = self.true_classes_documents.count(c)
            N_c = np.sum([self.true_classes_documents.count(m.id) for m in mesh_and_child_classes]) #+ all documents from all children of c
            N = len(self.true_classes_documents)
            #print("\nc: {}, N_c: {}, N: {}".format(meshid, N_c, N, N_c/N))
            #print("\t, mesh_childs: {}".format( mesh_and_child_classes))
            self.temp_max_doc_perClass_inCluster = 0
            if evaluateOnlyOnLeafs == False: # evaluate on all nodes
                _walk_F1_zhao(self.tree.children[0], mesh_and_child_classes, N_c, evaluateOnlyOnLeafs)
                _walk_F1_zhao(self.tree.children[1], mesh_and_child_classes, N_c, evaluateOnlyOnLeafs)
            else: # only leafs
                for leaf in leafs:
                    _walk_F1_zhao(leaf, mesh_and_child_classes, N_c, evaluateOnlyOnLeafs)
            #print("Best F1: {}".format(self.temp_max_doc_perClass_inCluster))
            FScore_sum += (N_c / N ) * self.temp_max_doc_perClass_inCluster
            #print("Score: {}".format((N_c / N ) * self.temp_max_doc_perClass_inCluster))

        return FScore_sum

    def get_isim(self, data):
        """ Internal similarity """

        I_sum = 0
        def _walk_isim(node):

            print("Node: {}".format(node))
            print("abstracts: {}".node.abstracts)
            for child in node.children:
                _walk_isim(child)



    def get_performances(self, evaluateOnlyOnLeafs=False):
        precision = self.get_precision()
        recall = self.get_recall()
        F1 = self.get_F1()
        return({
            "prec_micro" : precision["prec_micro"]
            ,"prec_macro" : precision["prec_macro"]
            ,"recall_micro" : recall["recall_micro"]
            ,"recall_macro" : recall["recall_macro"]
            ,"F1_micro" : F1["F1_micro"]
            ,"F1_macro" : F1["F1_macro"]
            ,"F1_zhao" : self.F1_zhao(evaluateOnlyOnLeafs=evaluateOnlyOnLeafs)
        })



class Node(object):
    "Generic tree node."
    def __init__(self, Id, depth, parent=None, cluster_label=None, children=[]):
        self.node_id = Id
        self.parent = parent
        self.children = []
        self.depth = depth
        self.cluster_label = cluster_label # In case FBE: this is the filterValue in the leafs
        self.abstracts = [] # PMID's of abstracts
        self.true_classes = [] # True classes for each abstract
        self.counts = 0
        self.recall = None
        self.precision = None
        self.F1 = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return "Node id: {} (depth: {}, cluster_label: {}, children: {})".format(
            self.node_id
            , self.depth
            , self.cluster_label
            , [child.node_id for child in self.children])

    def add_child(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

    def set_clusterLabel(self, clusterLabel):
        self.cluster_label = clusterLabel

    def pretty_print(self, depth=0):

        if self.depth == depth:
            print("Node: {}, Parent: {} (Depth: {}, counts: {}, cluster_label: {}) | Children: {}".format(self.node_id, self.parent, self.depth, self.counts, self.cluster_label, self.children))
            print("\tAbstracts: {}".format(Counter(self.abstracts)))
            print("\ttrue_classes: {}".format(Counter(self.true_classes)))
        else:
            for child in self.children:
                child.pretty_print(depth)


    def update_node(self, abstract_id, true_class):
        """ Updates the abstracts and its true class label running through this node """
        self.abstracts.append(abstract_id)
        self.true_classes.append(true_class)
        self.counts += 1


    def get_precision(self):
        count = Counter(self.true_classes)
        mostFrequent = max(self.true_classes, key=count.get)
        prec = self.true_classes.count(mostFrequent) / self.counts
        return prec

    def count_class_occurrence(self, c):
        return self.true_classes.count(c)


## Functions for FBE evaluation ##

def get_list_all_possible_classes(sentences, data ):
    """ Get the list of all possible occuring classes in the sentences parquet file """
    join_udf = udf(lambda x: ";".join(x))
    sentences_classes_udf = udf(lambda x: ";".join([str(v) for v in x.keys()]))

    sentences_transformed = sentences.select("id"
                                            , "tokens"
                                            , sentences_classes_udf('index').alias("all_classes")) \
                                    .withColumn("tokens", join_udf(col("tokens")))

    sentences_pdf = sentences_transformed.toPandas()
    sentences_pdf["id"] = pd.to_numeric(sentences_pdf["id"])

    # add true class labels to sentences from data by merge/join
    sentences_pdf["PMID"] = sentences_pdf["id"]
    sentences_pdf["PMID"] = pd.to_numeric(sentences_pdf["PMID"])
    meshDiab = data[["PMID", "mesh_ui_diab"]]
    meshDiab["PMID"] = pd.to_numeric(meshDiab["PMID"])
    sentences_pd_with_classes = pd.merge(sentences_pdf, meshDiab, on='PMID', how="left")


    # list of all classes in the sentences file
    return (set(pd.to_numeric(sentences_pdf["all_classes"].map(lambda sentence: sentence.split(";")).explode()).values)
            , sentences_pd_with_classes)



def matchCluster(index_map, cluster):
    """ gets for each document its unique cluster (filterValue) from the index"""
    return list(set(list(index_map.keys())).intersection(set(cluster)))[0]

def associate_unique_cluster_to_documents(sentences, tree):
    """ Associates unique cluster to each document """
    leafs = tree.get_leaf_nodes()
    print("N leafs: {}".format(len(leafs)))
    cluster = [leaf.cluster_label for leaf in leafs]
    print("N clusters: {}".format(len(set(cluster))))

    matchCluster_udf = udf(lambda y: matchCluster(y, cluster))
    join_udf = udf(lambda x: ";".join(x))

    sentences_transformed = sentences.select("id", "tokens", matchCluster_udf('index').alias("uniqueCluster")) \
        .withColumn("tokens", join_udf(col("tokens")))

    sentences_pd = sentences_transformed.toPandas() # spark dataframe -> pandas dataframe
    sentences_pd["id"] = pd.to_numeric(sentences_pd["id"])
    sentences_pd["uniqueCluster"] = pd.to_numeric(sentences_pd["uniqueCluster"])

    # add true class labels to data by merge/join
    sentences_pd["PMID"] = sentences_pd["id"]
    sentences_pd["PMID"] = pd.to_numeric(sentences_pd["PMID"])
    meshDiab = data[["PMID", "mesh_ui_diab"]]
    meshDiab["PMID"] = pd.to_numeric(meshDiab["PMID"])
    sentences_pd_with_classes_uniqueCluster = pd.merge(sentences_pd, meshDiab, on='PMID', how="left")

    return sentences_pd_with_classes_uniqueCluster





###### RUN ###########

spark = pyspark.sql.SparkSession.builder.getOrCreate()

data = pd.read_parquet(dataDir)
sentences = spark.read.load(modelDir+"/phrases/")
print("N sentences: {}".format(sentences.count()))

print("Load tree (nodes.json)..")
nodes = pd.read_json(modelDir+"/nodes.json", orient="records")
print("N nodes: {}".format(nodes.shape))

print("Get list with all possible classes in the sentences file..")
sentences_all_classes, sentences_pd_with_classes = get_list_all_possible_classes(sentences, data)
print("Number of classes in sentences file: {}".format(len(sentences_all_classes)))

print("initialise tree..")
treeFBE = Tree(nodes
            , mode="FBE"
            , sentences_all_classes=sentences_all_classes
            , true_classes_all=sentences_pd_with_classes["mesh_ui_diab"])

root = Node(Id=1, depth=0, parent=None, children=[]) # Id = 1 because start at Explorer
treeFBE.set_build_tree(root)

print("Associate cluster to each sentence..")
sentences_pd_with_classes_uniqueCluster = associate_unique_cluster_to_documents(sentences, treeFBE)
print("Unique clusters in sentences: {}".format(sentences_pd_with_classes_uniqueCluster["uniqueCluster"].nunique())) #####

print("Fit (only up to {} leafs)..".format(MAX_LEAFS))
treeFBE.fitTree(treeFBE.tree, sentences_pd_with_classes_uniqueCluster)
print("Get performances..")
pprint(treeFBE.get_performances(evaluateOnlyOnLeafs=False)["F1_Zhao"])
