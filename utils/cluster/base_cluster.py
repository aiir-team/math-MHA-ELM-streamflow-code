#!/usr/bin/env python
# Created by "Thieu" at 20:26, 06/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn import metrics
from copy import deepcopy
from scipy.spatial.distance import cdist


class BaseCluster:
    """
    https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    """

    def __init__(self):
        """
            n_clusters: the total number of clusters
            centers: the numpy array includes the clusters center
            list_clusters: the python array includes list of clusters center and the number of point belong to that cluster
            labels: the numpy array indicates which cluster that point belong to (based on index of array)
            feature_label: the concatenate of dataset and 1 column includes labels above
        """
        self.filename, self.paras, self.max_cluster = None, None, None
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = None, None, None, None, None

    def evaluation(self, X_data=None, labels=None):
        """
        :param X_data:
        :param labels: labels (predicted labels)
        :param type of score: 0 - all, 1 - silhouette, 2 - calinski harabaz, 3 - davies bouldin
        :return:
        """
        labels = np.ravel(labels)  # (1-d)
        silhouette = round(metrics.silhouette_score(X_data, labels, metric='euclidean'), 3)
        calinski = round(metrics.calinski_harabasz_score(X_data, labels), 3)
        davies = round(metrics.davies_bouldin_score(X_data, labels), 3)
        return {
            "silhouette": silhouette,
            "calinski": calinski,
            "davies": davies
        }

    def get_mutate_vector(self, vectorA=None, vectorB=None, mutation_type="mean"):
        """
        mutation_type: mean, uniform
        """
        vectorA, vectorB = np.array(vectorA), np.array(vectorB)
        offspring = np.zeros(np.array(vectorA).shape)
        if mutation_type == "mean":  # Mean
            offspring = (vectorA + vectorB) / 2.0
        elif mutation_type == "uniform":  # Uniform random point
            offspring = np.random.uniform(vectorA, vectorB)
        return offspring

    def cluster_covering(self, X_data=None, distance_level=None, mutation_type="mean"):
        """
        This method is inspired by immune algorithm in Phase2 Clustering of SONIA network
        :param X_data:
        :param list_clusters:
        :param distance_level:
        :param mutation_id: 0-Mean(parents), 1-Uniform(parents)
        :return:
            1st: the number of clusters
            2nd: the numpy array includes the clusters center
            3rd: the python array includes list of clusters center and the number of point belong to that cluster
            4th: the numpy array indicates which cluster that point belong to (based on index of array)
            5th: the concatenate of dataset and 1 column includes 4th
        """
        ## Adaptive distance level
        dis_level = round(distance_level * np.sqrt(X_data.shape[1]), 2)

        threshold_number = int(len(X_data) / len(self.list_clusters))
        centers = np.array([center[1] for center in self.list_clusters])
        ### Phrase 2: Mutated hidden unit - Adding artificial local data
        # Adding 2 hidden unit in beginning and ending points of input space
        t1 = np.zeros(X_data.shape[1])
        t2 = np.ones(X_data.shape[1])

        self.list_clusters.append([0, t1])
        self.list_clusters.append([0, t2])
        centers = np.concatenate((centers, np.array([t1])), axis=0)
        centers = np.concatenate((centers, np.array([t2])), axis=0)

        # Sort matrix weights input and hidden, Sort list hidden unit by list weights
        for i in range(0, centers.shape[1]):
            centers = sorted(centers, key=lambda elem_list: elem_list[i])
            self.list_clusters = sorted(self.list_clusters, key=lambda elem_list: elem_list[1][i])

            for i in range(len(self.list_clusters) - 1):
                ta, wHa = self.list_clusters[i][0], self.list_clusters[i][1]
                tb, wHb = self.list_clusters[i + 1][0], self.list_clusters[i + 1][1]
                dab = np.linalg.norm(wHa - wHb)  # Distance function

                if dab > dis_level and ta < threshold_number and tb < threshold_number:
                    threshold_number = int(len(X_data) / len(self.list_clusters))  # Re-calculate threshold
                    # Create new mutated hidden unit
                    temp_node = self.get_mutate_vector(wHa, wHb, mutation_type)
                    self.list_clusters.insert(i + 1, [0, deepcopy(temp_node)])
                    centers = np.insert(centers, [i + 1], deepcopy(temp_node), axis=0)
            #         print "New hidden unit created. {0}".format(len(self.matrix_Wih))
            # print("Finished mutation hidden unit!!!")

        ## Now: time to re-calculate which cluster a point belong to?
        labels = np.zeros(len(X_data))
        for i in range(0, len(X_data)):
            list_dist_mj = []  # list dist(mj)
            for j in range(0, len(self.list_clusters)):  # j: index of hidden unit jth
                list_dist_mj.append([j, np.linalg.norm(X_data[i] - centers[j])])
            list_dist_mj = sorted(list_dist_mj, key=lambda item: item[1])  ## sort
            labels[i] = list_dist_mj[0][0]  # c: Index of hidden unit cth which make the distance minimum

        self.n_clusters = len(self.list_clusters)
        self.centers = deepcopy(centers)
        self.labels = deepcopy(np.reshape(labels, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels, (-1, 1))), axis=1)

    def cluster_scattering(self, X_data=None):
        ## Phase 1 can use Kmean, Immune, Mean Shift, DBScan, ...
        pass

    def transform(self, X_data=None, function=None):
        return function(cdist(X_data, self.centers))
