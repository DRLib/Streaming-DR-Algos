#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import numpy as np
from numba import jit


class DataRepo:
    def __init__(self, n_neighbor):
        self.n_neighbor = n_neighbor
        self._total_n_samples = 0
        self._total_data = None
        self._total_label = None
        self._total_embeddings = None
        self._knn_manager = KNNManager(n_neighbor)

    def slide_window(self, out_num, *args):
        if out_num <= 0:
            return out_num

        self._total_data = self.get_total_data()[out_num:]
        self._total_label = self.get_total_label()[out_num:]
        self._total_embeddings = self.get_total_embeddings()[out_num:]
        self._knn_manager.slide_window(out_num)
        self._total_n_samples = self._total_data.shape[0]

        return out_num

    def post_slide(self, out_num):
        self._total_embeddings = self.get_total_embeddings()[out_num:]

    def get_n_samples(self):
        return self._total_n_samples

    def get_total_data(self):
        return self._total_data

    def get_total_label(self):
        return self._total_label

    def get_total_embeddings(self):
        return self._total_embeddings

    def get_knn_indices(self):
        return self._knn_manager.knn_indices

    def get_knn_dists(self):
        return self._knn_manager.knn_dists

    def update_embeddings(self, new_embeddings):
        self._total_embeddings = new_embeddings

    def add_new_data(self, data=None, embeddings=None, labels=None, knn_indices=None, knn_dists=None):
        sta = time.time()
        if data is not None:
            if self._total_data is None:
                self._total_data = data
            else:
                # self._total_data = np.concatenate([self._total_data, data], axis=0)
                self._total_data = np.append(self._total_data, data, axis=0)
            self._total_n_samples += data.shape[0]
        # print("add data:", time.time() - sta)

        if embeddings is not None:
            if self._total_embeddings is None:
                self._total_embeddings = embeddings
            else:
                self._total_embeddings = np.concatenate([self._total_embeddings, embeddings], axis=0)

        if labels is not None:
            if self._total_label is None:
                self._total_label = np.array(labels)
            else:
                if isinstance(labels, list):
                    self._total_label = np.concatenate([self._total_label, labels])
                else:
                    self._total_label = np.append(self._total_label, labels)

        if knn_indices is not None and knn_dists is not None:
            self._knn_manager.add_new_kNN(knn_indices, knn_dists)


class KNNManager:
    def __init__(self, k):
        self.k = k
        self.knn_indices = None
        self.knn_dists = None
        self._pre_neighbor_changed_meta = []

    def update_knn_graph(self, knn_indices, knn_dists):
        self.knn_indices = knn_indices
        self.knn_dists = knn_dists

    def slide_window(self, out_num):
        if self.knn_indices is None or out_num <= 0:
            return

        self.knn_indices = self.knn_indices[out_num:]
        self.knn_dists = self.knn_dists[out_num:]

    def is_empty(self):
        return self.knn_indices is None

    def get_pre_neighbor_changed_positions(self):
        return self._pre_neighbor_changed_meta

    def add_new_kNN(self, new_knn_indices, new_knn_dists):
        if self.knn_indices is None:
            self.knn_indices = new_knn_indices
            self.knn_dists = new_knn_dists
            return

        if new_knn_indices is not None:
            self.knn_indices = np.concatenate([self.knn_indices, new_knn_indices], axis=0)
        if new_knn_dists is not None:
            self.knn_dists = np.concatenate([self.knn_dists, new_knn_dists], axis=0)

    def update_previous_kNN(self, new_data_num, pre_n_samples, dists2pre_data, data_num_list=None,
                            neighbor_changed_indices=None, symm=True):
        self._pre_neighbor_changed_meta = []
        farest_neighbor_dist = self.knn_dists[:, -1]

        if neighbor_changed_indices is None:
            neighbor_changed_indices = []

        tmp_index = 0
        for i in range(new_data_num):
            indices = np.where(dists2pre_data[i] < farest_neighbor_dist)[0]
            flag = True

            if data_num_list is not None:
                if i > 0 and i == data_num_list[tmp_index + 1]:
                    tmp_index += 1
                indices = indices[np.where(indices < data_num_list[tmp_index] + pre_n_samples)[0]]

                if len(indices) < 1:
                    flag = False
            else:
                indices = np.setdiff1d(indices, [pre_n_samples + i])

            if flag:
                for j in indices:
                    if j not in neighbor_changed_indices:
                        neighbor_changed_indices.append(j)
                    insert_index = self.knn_dists.shape[1] - 1
                    while insert_index >= 0 and dists2pre_data[i][j] <= self.knn_dists[j][insert_index]:
                        insert_index -= 1

                    if symm and self.knn_indices[j][-1] not in neighbor_changed_indices:
                        neighbor_changed_indices.append(self.knn_indices[j][-1])

                    self._pre_neighbor_changed_meta.append(
                        [i + pre_n_samples, j, insert_index + 1, self.knn_indices[j][-1]])
                    arr_move_one(self.knn_dists[j], insert_index + 1, dists2pre_data[i][j])
                    arr_move_one(self.knn_indices[j], insert_index + 1, pre_n_samples + i)
                    farest_neighbor_dist[j] = self.knn_dists[j, -1]

        self._pre_neighbor_changed_meta = np.array(self._pre_neighbor_changed_meta, dtype=int)

        return neighbor_changed_indices

    def update_previous_kNN_simple(self, new_data_num, pre_n_samples, candidate_indices, candidate_dists,
                                   neighbor_changed_indices=None, symm=True):
        neighbor_changed_indices, self._pre_neighbor_changed_meta, self.knn_indices, self.knn_dists = \
            _do_update(new_data_num, pre_n_samples, candidate_indices, candidate_dists, self.knn_indices, self.knn_dists,
                       neighbor_changed_indices, symm)
        self._pre_neighbor_changed_meta = np.array(self._pre_neighbor_changed_meta, dtype=int)

        return neighbor_changed_indices


@jit(nopython=True)
def _do_update(new_data_num, pre_n_samples, candidate_indices_list, candidate_dists_list, knn_indices, knn_dists,
               neighbor_changed_indices, symm=True):
    pre_neighbor_changed_meta = []

    for i in range(new_data_num):
        candidate_indices = candidate_indices_list[i]
        candidate_dists = candidate_dists_list[i]

        for j, data_idx in enumerate(candidate_indices):
            if knn_dists[data_idx][-1] <= candidate_dists[j]:
                continue
            # data_idx = candidate_indices[j]
            if data_idx not in neighbor_changed_indices:
                neighbor_changed_indices.append(data_idx)

            insert_index = knn_dists.shape[1] - 1
            while insert_index >= 0 and candidate_dists[j] <= knn_dists[data_idx][insert_index]:
                insert_index -= 1

            if symm and knn_indices[data_idx][-1] not in neighbor_changed_indices:
                neighbor_changed_indices.append(knn_indices[data_idx][-1])

            pre_neighbor_changed_meta.append(
                [pre_n_samples + i, data_idx, insert_index + 1, knn_indices[data_idx][-1]])

            knn_dists[data_idx][insert_index + 2:] = knn_dists[data_idx][insert_index + 1:-1]
            knn_dists[data_idx][insert_index + 1] = candidate_dists[j]

            knn_indices[data_idx][insert_index + 2:] = knn_indices[data_idx][insert_index + 1:-1]
            knn_indices[data_idx][insert_index + 1] = pre_n_samples

    return neighbor_changed_indices, pre_neighbor_changed_meta, knn_indices, knn_dists


# @jit
def extract_csr(csr_graph, indices, norm=True):
    nn_indices = []
    nn_weights = []

    for i in indices:
        pre = csr_graph.indptr[i]
        idx = csr_graph.indptr[i + 1]
        cur_indices = csr_graph.indices[pre:idx]
        cur_weights = csr_graph.data[pre:idx]

        nn_indices.append(cur_indices)
        if norm:
            nn_weights.append(cur_weights / np.sum(cur_weights))
        else:
            nn_weights.append(cur_weights)
    return nn_indices, nn_weights


def arr_move_one(arr, index, index_val):
    arr[index + 1:] = arr[index:-1]
    arr[index] = index_val
