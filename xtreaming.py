import time

import numpy as np
from procrustes import orthogonal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

from upd import cal_dist, UPDis4Streaming


def sampled_control_points(data):
    n_samples = data.shape[0]
    km = KMeans(n_clusters=int(np.sqrt(n_samples)))
    km.fit(data)
    centroids = km.cluster_centers_
    dists = cdist(centroids, data)

    sampled_indices = np.argsort(dists, axis=-1)[:, 0]
    sampled_x = data[sampled_indices]
    return sampled_x, sampled_indices


def procrustes_analysis(pre_data, cur_data, all_data, align=True, scale=True, translate=True):
    if not align:
        return all_data

    result = orthogonal(cur_data, pre_data, scale=scale, translate=translate)

    aligned_embeddings = np.dot(all_data, result.t)
    return aligned_embeddings


class XtreamingModel:
    def __init__(self, buffer_size=200, eta=0.99):
        self.buffer_size = buffer_size
        self._eta = eta
        self.pro_model = UPDis4Streaming(eta)
        self.buffered_data = None
        self.time_step = 0
        self.pre_control_indices = None
        self.pre_control_points = None
        self.pre_embedding = None
        self.pre_cntp_embeddings = None
        self.lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean", contamination=0.1)
        self.time_costs = 0
        self._time_cost_records = [0]
        self._total_n_samples = 0

        self.total_data = None

    def _buffering(self, data):
        if self.buffered_data is None:
            self.buffered_data = data
        else:
            self.buffered_data = np.concatenate([self.buffered_data, data], axis=0)

        self.time_step += 1
        return self.buffered_data.shape[0] >= self.buffer_size

    def fit_new_data(self, data, labels=None):
        key_time = 0
        sta = time.time()
        if not self._buffering(data):
            return self.pre_embedding, 0, False, 0

        self.total_data = self.buffered_data if self.total_data is None else np.concatenate([self.total_data,
                                                                                             self.buffered_data], axis=0)
        self._total_n_samples += self.buffered_data.shape[0]

        ret = self.fit()
        return ret

    def buffer_empty(self):
        return self.buffered_data is None

    def fit(self):
        medoids, sampled_indices = sampled_control_points(self.buffered_data)
        if self.pre_embedding is None:
            self._initial_project(medoids, sampled_indices)
        else:
            cur_control_points, drift_indices = self._detect_concept_drift(medoids)
            cur_control_indices = sampled_indices[drift_indices]
            dists2cntp = cdist(self.buffered_data, self.pre_control_points)

            if cur_control_points is None:
                cur_embeddings = self.pro_model.reuse_project(dists2cntp)
                self.pre_embedding = np.concatenate([self.pre_embedding, cur_embeddings], axis=0)
            else:
                aligned_total_embeddings, total_cntp_points = self._re_projection(dists2cntp, cur_control_indices,
                                                                                  cur_control_points)
                self.pre_control_points = total_cntp_points
                self.pre_embedding = aligned_total_embeddings
                self.lof.fit(self.pre_control_points)

        self.buffered_data = None

        return self.pre_embedding

    def _initial_project(self, medoids, sampled_indices):
        dists = cdist(self.buffered_data, self.buffered_data)
        cntp2cntp_dists = dists[sampled_indices, :][:, sampled_indices]
        dists2cntp = dists[:, sampled_indices]
        self.pre_control_points = medoids
        self.pre_control_indices = sampled_indices
        self.pre_embedding = self.pro_model.fit_transform(dists2cntp, cntp2cntp_dists)
        self.pre_cntp_embeddings = self.pre_embedding[sampled_indices]
        self.buffered_data = None
        self.lof.fit(self.pre_control_points)

    def _detect_concept_drift(self, cur_medoids):
        labels = self.lof.predict(cur_medoids)
        control_indices = np.where(labels == -1)[0]
        cur_control_points = None if len(control_indices) == 0 else cur_medoids[control_indices.astype(int)]
        return cur_control_points, control_indices

    def _re_projection(self, dists2cntp, control_indices, control_points):

        pre_data_num = self.pre_embedding.shape[0]
        total_cntp_points = np.concatenate([self.pre_control_points, control_points], axis=0)

        dists2pre_cntp = cdist(self.pre_embedding, self.pre_cntp_embeddings)
        cur_cntp_embeddings = self.pro_model.reuse_project(dists2cntp[control_indices])
        dists2new_cntp = cdist(self.pre_embedding, cur_cntp_embeddings)
        pre_dists2all_cntp = np.concatenate([dists2pre_cntp, dists2new_cntp], axis=1)

        total_cntp_embeddings = np.concatenate([self.pre_cntp_embeddings, cur_cntp_embeddings], axis=0)
        cntp_dists = cal_dist(total_cntp_embeddings)
        updated_pre_embeddings = self.pro_model.fit_transform(pre_dists2all_cntp, cntp_dists)
        new_embeddings = self.pro_model.reuse_project(cdist(self.buffered_data, total_cntp_points))
        total_embeddings = np.concatenate([updated_pre_embeddings, new_embeddings], axis=0)

        updated_pre_cntp_embeddings = self.pro_model.reuse_project(cdist(self.pre_control_points, total_cntp_points))
        aligned_total_embeddings = procrustes_analysis(self.pre_cntp_embeddings,
                                                       updated_pre_cntp_embeddings, total_embeddings, align=True)

        self.pre_control_indices = np.concatenate([self.pre_control_indices, control_indices + pre_data_num])
        return aligned_total_embeddings, total_cntp_points

    def transform(self, data):
        dists = cdist(data, self.pre_control_points)
        embeddings = self.pro_model.reuse_project(dists)
        return embeddings
