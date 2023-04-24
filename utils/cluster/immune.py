#!/usr/bin/env python
# Created by "Thieu" at 20:26, 06/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from utils.cluster.base_cluster import BaseCluster
from scipy.spatial.distance import cdist
from copy import deepcopy
import numpy as np


class ImmuneInspiration(BaseCluster):
    """
    This cluster algorithm implement based on from paper:
        Improving recognition and generalization capability of
        back-propagation NN using a self-organized network inspired by immune algorithm (SONIA)
    """

    def __init__(self, para_dict, **kwargs):
        """
        :param stimulation_level:
        :param positive_number:
        :param distance_level:
        :param mutation_id: 0 - Mean(parents), 1 - Uniform(parents)
        :param max_cluster:
        """
        super().__init__()
        self.kernel = "IM"
        self.stimulation_level = 0.25
        self.positive_number = 0.15
        self.distance_level = 0.5
        self.max_cluster = 50
        self.mutation_type = "mean"
        if not (para_dict is None) and (type(para_dict) == dict):
            self.__set_keyword_arguments(para_dict)
        if kwargs is not None:
            self.__set_keyword_arguments(kwargs)

        self.paras = {"kernel": self.kernel, "stimulation_level": self.stimulation_level,
                      "positive_number": self.positive_number, "max_cluster": self.max_cluster}
        self.filename = f"{self.kernel}-{self.stimulation_level}-{self.positive_number}-{self.max_cluster}"
        if self.kernel == "IMF":
            self.paras["distance_level"] = self.distance_level
            self.paras["mutation_type"] = self.mutation_type
            self.filename = f"{self.filename}-{self.distance_level}-{self.mutation_type}"

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def clustering(self, X_data=None):
        if self.kernel == "IM":
            self.cluster_scattering(X_data)
        elif self.kernel == "IMF":
            self.cluster_scattering(X_data)
            self.cluster_covering(X_data, self.distance_level, self.mutation_type)

    # def clustering(self, kernel="IMF", X_data=None, distance_level=None, mutation_type="mean"):
    #     if kernel == "IM":
    #         self.cluster_scattering(X_data)
    #     elif kernel == "IMF":
    #         self.cluster_scattering(X_data)
    #         self.cluster_covering(X_data, distance_level, mutation_type)

    def cluster_scattering(self, X_data=None):
        """
        :param X_data:
        :return:
        1st: the number of clusters
        2nd: the numpy array includes the clusters center
        3rd: the python array includes list of clusters center and the number of point belong to that cluster
        4th: the numpy array indicates which cluster that point belong to (based on index of array)
        5th: the concatenate of dataset and 1 column includes 4th
        """
        ### Phrase 1: Cluster data - First step training of two-step training
        # 2. Init hidden unit 1st
        ## Adaptive stimulation level
        sti_level = round(self.stimulation_level * np.sqrt(X_data.shape[1]), 2)

        y_pred = np.zeros(len(X_data))
        hu1 = [0, deepcopy(X_data[np.random.randint(0, len(X_data))])]  # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]  # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])  # 2-d array
        m = 0
        while m < len(X_data):
            D = cdist(np.reshape(X_data[m], (1, -1)), centers)  # calculate pairwise distances btw mth point and centers
            c = np.argmin(D, axis=1)[0]  # return index of the closest center
            distmc = np.min(D, axis=1)  # distmc: minimum distance

            if distmc < sti_level:
                y_pred[m] = c  # Which cluster c, does example m belong to?
                list_clusters[c][0] += 1  # update hidden unit cth

                centers[c] = centers[c] + self.positive_number * distmc * (X_data[m] - list_clusters[c][1])
                list_clusters[c][1] = list_clusters[c][1] + self.positive_number * distmc * (X_data[m] - list_clusters[c][1])
                # Next example
                m += 1
                # if m % 1000 == 0:
                #     print "distmc = {0}".format(distmc)
                #     print "m = {0}".format(m)
            else:
                # print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(X_data[m])])
                # print "Hidden unit: {0} created.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(X_data[m])], axis=0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    print(f"Clustering Stopped: Over the number of clusters allowable: {self.max_cluster}")
                    break
        self.n_clusters = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)
        self.labels = deepcopy(np.reshape(y_pred, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(y_pred, (-1, 1))), axis=1)
