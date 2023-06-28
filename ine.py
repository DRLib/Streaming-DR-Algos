import time

import h5py
import numpy as np
import scipy.optimize
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE
from utils.nn_utils import compute_knn_graph
from utils.warppers import KNNManager, DataRepo


def tsne_grad(center_embedding, high_sims, neighbor_embeddings, k):
    center_embedding = center_embedding[np.newaxis, :]
    dists = cdist(center_embedding, neighbor_embeddings) ** 2
    a = (1 + dists) ** -1
    f = 1 / (1 + dists) ** (0.5 + k / 10)
    q = f / np.sum(f)
    grad = (1 + k / 5) * np.sum(a * (high_sims - q) * (center_embedding - neighbor_embeddings).T, axis=1)
    return grad


class kNNBasedIncrementalMethods:
    def __init__(self, train_num, n_components, n_neighbors, single=False):
        self.single = single
        self.initial_train_num = train_num
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.knn_manager = KNNManager(n_neighbors)
        self.stream_dataset = DataRepo(n_neighbors)
        self.pre_embeddings = None
        self.trained = False
        self.time_cost = 0
        self._time_cost_records = [0]

    def _update_kNN(self, new_data):
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)

        new_data_num = new_data.shape[0]
        pre_data_num = self.stream_dataset.get_n_samples() - new_data_num
        neighbor_changed_indices = self.knn_manager.update_previous_kNN(new_data_num, pre_data_num, dists,
                                                                        data_num_list=[0, new_data_num])
        neighbor_changed_indices = np.array(neighbor_changed_indices)
        return knn_indices, knn_dists, neighbor_changed_indices[neighbor_changed_indices < pre_data_num]

    def _cal_new_data_kNN(self, new_data, include_self=True):
        new_data_num = new_data.shape[0]
        if include_self:
            data = self.stream_dataset.get_total_data()
        else:
            data = self.stream_dataset.get_total_data()[:-1]
        dists = cdist(new_data, data)
        knn_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        knn_dists = np.zeros(shape=(new_data_num, self.n_neighbors))

        for i in range(new_data_num):
            knn_dists[i] = dists[i][knn_indices[i]]
        return knn_indices, knn_dists, dists

    def _first_train(self, train_data):
        pass

    def fit_new_data(self, x, labels=None):
        if self.single:
            return self._fit_new_data_single(x, labels)
        else:
            return self._fit_new_data_batch(x, labels)

    def _fit_new_data_batch(self, x, labels=None):
        self.stream_dataset.add_new_data(x, None, labels)

        if not self.trained:
            if self.stream_dataset.get_n_samples() < self.initial_train_num:
                return None
            self.trained = True
            self._first_train(self.stream_dataset.get_total_data())
        else:
            self._incremental_embedding(x)

        return self.pre_embeddings

    def _fit_new_data_single(self, x, labels=None):
        key_time = 0
        if not self.trained:
            self.stream_dataset.add_new_data(x, None, labels)
            sta = time.time()
            if self.stream_dataset.get_n_samples() >= self.initial_train_num:
                self.trained = True
                self._first_train(self.stream_dataset.get_total_data())
        else:
            for i, item in enumerate(x):
                self.stream_dataset.add_new_data(np.reshape(item, (1, -1)), None, labels[i] if labels is not None else None)
                self._incremental_embedding(np.reshape(item, (1, -1)))

        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        pass

    def ending(self):
        output = "Time Cost: %.4f" % self.time_cost
        print(output)
        return output, self._time_cost_records


def _select_min_loss_one(candidate_embeddings, neighbors_embeddings, high_probabilities, k):
    if len(neighbors_embeddings.shape) < 2:
        neighbors_embeddings = np.reshape(neighbors_embeddings, (1, -1))
    if len(neighbors_embeddings.shape) > 2:
        neighbors_embeddings = np.squeeze(neighbors_embeddings)
    dists = cdist(candidate_embeddings, neighbors_embeddings)
    tmp_prob = 1 / (1 + dists ** 2) ** (0.5 + k/10)
    q = tmp_prob / np.expand_dims(np.sum(tmp_prob, axis=1), axis=1)
    high_prob_matrix = np.repeat(high_probabilities, candidate_embeddings.shape[0], axis=0)
    loss_list = -np.sum(high_prob_matrix * np.log(q), axis=1)
    return candidate_embeddings[np.argmin(loss_list)]


class INEModel(kNNBasedIncrementalMethods, TSNE):
    def __init__(self, train_num, n_components, n_neighbors, iter_num=100, grid_num=27, desired_perplexity=3,
                 init="random"):
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors, True)
        TSNE.__init__(self, n_components, perplexity=n_neighbors)
        self.init = init
        self.desired_perplexity = desired_perplexity
        self.iter_num = iter_num
        self.grid_num = grid_num
        self.condition_P = None
        self._learning_rate = 200.0
        self._update_thresh = 5
        self._k = 5

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        knn_indices, knn_dists = compute_knn_graph(train_data, None, self.n_neighbors, None)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        new_data = np.reshape(new_data, (1, -1))

        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)
        knn_indices = knn_indices.squeeze()

        new_data_prob = self._cal_new_data_probability(knn_dists.astype(np.float32, copy=False))

        initial_embedding = self._initialize_new_data_embedding(new_data_prob, knn_indices)
        self.pre_embeddings = self._optimize_new_data_embedding(knn_indices, initial_embedding, new_data_prob)
        return self.pre_embeddings

    def _cal_new_data_probability(self, dists):
        conditional_P = search_prob(dists**2, perplexity=self.desired_perplexity)
        self.condition_P = np.concatenate([self.condition_P, conditional_P], axis=0)
        return conditional_P

    def _initialize_new_data_embedding(self, high_prob, new_knn_indices):
        candidate_embeddings = self._generate_candidate_embeddings()
        initial_embeddings = _select_min_loss_one(candidate_embeddings, self.pre_embeddings[new_knn_indices],
                                                  high_prob, self._k)
        initial_embeddings = np.mean(self.pre_embeddings[new_knn_indices], axis=0)
        return initial_embeddings

    def _optimize_new_data_embedding(self, new_data_knn_indices, initial_embedding, new_data_prob):
        def loss_func(embeddings, high_prob, neighbor_embeddings, k):
            similarities = 1 / (1 + cdist(embeddings[np.newaxis, :], neighbor_embeddings) ** 2) ** (0.5 + k/10)
            normed_similarities = similarities / np.expand_dims(np.sum(similarities, axis=1), axis=1)
            return -np.sum(high_prob * np.log(normed_similarities))

        res = scipy.optimize.minimize(loss_func, initial_embedding, method="Newton-CG", jac=tsne_grad,
                                      args=(new_data_prob, self.pre_embeddings[new_data_knn_indices], self._k),
                                      options={'gtol': 1e-6, 'disp': False})
        if np.abs(res.x[0] - initial_embedding[0]) > self._update_thresh \
                or np.abs(res.x[1] - initial_embedding[1]) > self._update_thresh:
            new_embeddings = initial_embedding[np.newaxis, :]
        else:
            new_embeddings = res.x[np.newaxis, :]
        total_embeddings = np.concatenate([self.pre_embeddings, new_embeddings], axis=0)
        return total_embeddings

    def _generate_candidate_embeddings(self):
        x_min, y_min = np.min(self.pre_embeddings, axis=0)
        x_max, y_max = np.max(self.pre_embeddings, axis=0)
        x_grid_list = np.linspace(x_min, x_max, self.grid_num)
        y_grid_list = np.linspace(y_min, y_max, self.grid_num)

        x_grid_list_ravel = np.reshape(np.repeat(np.expand_dims(x_grid_list, axis=1), self.grid_num, 1), (-1, 1))
        y_grid_list_ravel = np.reshape(np.repeat(np.expand_dims(y_grid_list, axis=0), self.grid_num, 0), (-1, 1))
        candidate_embeddings = np.concatenate([x_grid_list_ravel, y_grid_list_ravel], axis=1)
        return candidate_embeddings

    def _fit(self, X, skip_num_points=0):
        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            raise RuntimeError("Only support Barnes Hut t-SNE")
        else:
            n_neighbors = min(n_samples - 1, int(self.perplexity))
            knn = NearestNeighbors(
                algorithm="auto",
                n_jobs=self.n_jobs,
                n_neighbors=n_neighbors,
                metric=self.metric,
            )
            knn.fit(X)
            distances_nn = knn.kneighbors_graph(mode="distance")
            del knn

            if self.square_distances is True or self.metric == "euclidean":
                distances_nn.data **= 2

            P, conditional_P = cu_joint_probabilities_nn(distances_nn, self.perplexity)
            self.condition_P = conditional_P

        if self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=self.random_state,
            )
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == "random":
            random_state = check_random_state(self.random_state)
            X_embedded = 1e-4 * random_state.randn(n_samples, self.n_components).astype(
                np.float32
            )
        else:
            raise ValueError("'init' must be 'pca', 'random', or a numpy array")

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )


def cu_joint_probabilities_nn(distances, desired_perplexity):
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(
        distances_data, desired_perplexity, False
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    return P, conditional_P


def cal_perplexity(dist, idx=0, beta=1.0):
    prob = np.exp(-dist * beta)
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perplexity = -12
    else:
        prob /= sum_prob
        perplexity = 0
        for pj in prob:
            if pj != 0:
                perplexity += -pj * np.log(pj)
    return perplexity, prob


def search_prob(distances, tol=1e-5, perplexity=30.0, debug=False):
    (n, d) = distances.shape

    pair_prob = np.zeros_like(distances)
    beta = np.ones((n, 1))
    base_perplexity = np.log(perplexity)

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        perplexity, cur_prob = cal_perplexity(distances[i], i, beta[i])
        perplexity_diff = perplexity - base_perplexity
        tries = 0
        while np.abs(perplexity_diff) > tol and tries < 50:
            if perplexity_diff > 0:
                beta_min = beta[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + beta_max) / 2
            else:
                beta_max = beta[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + beta_min) / 2

            perplexity, cur_prob = cal_perplexity(distances[i], i, beta[i])
            perplexity_diff = perplexity - base_perplexity
            tries = tries + 1

        pair_prob[i, ] = cur_prob

    return pair_prob