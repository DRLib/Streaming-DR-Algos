import os
import time

import numba
import numpy as np
from pynndescent import NNDescent
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


def compute_accurate_knn(flattened_data, k, neighbors_cache_path=None, pairwise_cache_path=None, metric="euclidean",
                         include_self=False):
    cur_path = None
    if neighbors_cache_path is not None:
        cur_path = neighbors_cache_path

    if cur_path is not None and os.path.exists(cur_path):
        knn_indices, knn_distances = np.load(cur_path)
    else:
        preload = flattened_data.shape[0] <= 30000

        pairwise_distance = get_pairwise_distance(flattened_data, metric, pairwise_cache_path, preload=preload)
        sorted_indices = np.argsort(pairwise_distance, axis=1)
        if include_self:
            knn_indices = sorted_indices[:, :k]
        else:
            knn_indices = sorted_indices[:, 1:k + 1]
        knn_distances = []
        for i in range(knn_indices.shape[0]):
            knn_distances.append(pairwise_distance[i, knn_indices[i]])
        knn_distances = np.array(knn_distances)
        if cur_path is not None:
            np.save(cur_path, [knn_indices, knn_distances])

    return knn_indices, knn_distances


def compute_knn_graph(all_data, neighbors_cache_path, k, pairwise_cache_path,
                      metric="euclidean", max_candidates=60, accelerate=False, include_self=False):
    flattened_data = all_data.reshape((len(all_data), np.product(all_data.shape[1:])))
    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(flattened_data, k, neighbors_cache_path, pairwise_cache_path,
                                                          include_self=include_self)
        return knn_indices, knn_distances

    if neighbors_cache_path is not None and os.path.exists(neighbors_cache_path):
        neighbor_graph = np.load(neighbors_cache_path)
        knn_indices, knn_distances = neighbor_graph
    else:
        # number of trees in random projection forest
        n_trees = 5 + int(round((all_data.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(all_data.shape[0]))))

        nnd = NNDescent(
            flattened_data,
            n_neighbors=k + 1,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=max_candidates,
            verbose=False
        )
        knn_indices, knn_distances = nnd.neighbor_graph
        knn_indices = knn_indices[:, 1:]
        knn_distances = knn_distances[:, 1:]
        if neighbors_cache_path is not None:
            np.save(neighbors_cache_path, [knn_indices, knn_distances])
    return knn_indices, knn_distances


def get_pairwise_distance(flattened_data, metric="euclidean", pairwise_distance_cache_path=None, preload=False):
    if pairwise_distance_cache_path is not None and preload and os.path.exists(pairwise_distance_cache_path):
        pairwise_distance = np.load(pairwise_distance_cache_path)
    else:
        pairwise_distance = pairwise_distances(flattened_data, metric=metric, squared=False)
        pairwise_distance[pairwise_distance < 1e-12] = 0.0
        if preload and pairwise_distance_cache_path is not None:
            np.save(pairwise_distance_cache_path, pairwise_distance)
    return pairwise_distance
