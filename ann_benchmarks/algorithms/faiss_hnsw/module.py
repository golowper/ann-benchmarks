import faiss
import numpy as np

from ..faiss.module import Faiss

from sklearn.decomposition import PCA
import sklearn.preprocessing
import numpy


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param

    def fit(self, X):
        self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index.add(X)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p


class FaissHNSWPCA(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param

    def fit(self, X):
        self.pca = PCA(n_components=80)
        X_reduced = self.pca.fit_transform(X)
        
        self.index = faiss.IndexHNSWFlat(len(X_reduced[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == "angular":
            X_reduced = X_reduced / np.linalg.norm(X_reduced, axis=1)[:, np.newaxis]
        if X_reduced.dtype != np.float32:
            X_reduced = X_reduced.astype(np.float32)

        self.index.add(X_reduced)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef
    
    def batch_query(self, X, n):
        # print("X shape before:", X.shape)
        X = self.pca.transform(X)
        # print("X shape after:", X.shape)
        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def __str__(self):
        return "faiss-hnsw-pca (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p