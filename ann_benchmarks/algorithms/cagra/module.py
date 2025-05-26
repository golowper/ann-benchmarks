import numpy as np
import cupy as cp
from ..base.module import BaseANN
from cuvs.neighbors import cagra, hnsw


class CagraHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "inner_product", "euclidean": "speuclidean"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Fommat dataset
        if not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float32)
        elif X.dtype != cp.float32:
            X = X.astype(cp.float32)

        # Normalize the dataset if the metric is angular
        if self.metric == "inner_product":
            norms = cp.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)  # add epsilon to avoid division by zero
        
        # Build index
        index_params = cagra.IndexParams(
            metric= self.metric, 
            build_algo='nn_descent',
            nn_descent_niter=self.method_param.get("nnDescentNIter", 20),
            graph_degree=self.method_param["M"],
            intermediate_graph_degree=self.method_param.get("IntermediateGraphDegree", 128)
        )
        self.index = cagra.build(index_params, X)
        self.index = hnsw.from_cagra(hnsw.IndexParams(), self.index)

    def set_query_arguments(self, ef):
        self.ef_query = ef
        # Create search parameters (default ef and num_threads can be overridden in self.method_param)
        self.search_params = hnsw.SearchParams(
            ef=ef,
            num_threads=0
        )
        self.name = "Cagra HNSW (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        # Normalize the vector if the metric is angular
        if self.metric == "inner_product":
            v = cp.asarray(v) / cp.linalg.norm(v)
        if v.dtype != cp.float32:
            v = v.astype(cp.float32)
        # Expand dimensions to create a 2D array and convert to a numpy array
        v_host = cp.asnumpy(v[None, :])
        
        # Create search parameters (default ef and num_threads can be overridden in self.method_param)
        
        # Execute the search and get (distances, neighbors)
        distances, neighbors = hnsw.search(self.search_params, self.index, v_host, n)

        # Return the neighbor indices for the query
        return neighbors[0]

    def batch_query(self, X, n):
        # Normalize each vector if the metric is angular
        if self.metric == "inner_product":
            X = cp.asarray(X)
            norms = cp.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
        if X.dtype != cp.float32:
            X = X.astype(cp.float32)
        # Convert the batch queries to a numpy array
        X_host = cp.asnumpy(X)
        
        # Execute batch search and store the result in self.res
        self.res = hnsw.search(self.search_params, self.index, X_host, n)

    def get_batch_results(self):
        # Retrieve distances and neighbor indices from the batch search result
        distances, neighbours = self.res
        results = []
        # Filter out invalid indices (-1) for each query result
        for i in range(len(distances)):
            query_result = []
            for distance, neighbour in zip(distances[i], neighbours[i]):
                if neighbour != -1:
                    query_result.append(neighbour)
            results.append(query_result)
        return results

    def __str__(self):
        return "CagraHNSW M=%d nnDescentNIter=%d efQuery=%d" % (self.method_param["M"], self.method_param.get("nnDescentNIter", 20), self.ef_query)

    def freeIndex(self):
        del self.index


class Cagra(CagraHNSW):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "inner_product", "euclidean": "speuclidean"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Fommat dataset
        if not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float32)
        elif X.dtype != cp.float32:
            X = X.astype(cp.float32)

        # Normalize the dataset if the metric is angular
        if self.metric == "inner_product":
            norms = cp.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)
        
        # Build index
        index_params = cagra.IndexParams(
            metric= self.metric, 
            build_algo='nn_descent',
            nn_descent_niter=self.method_param.get("nnDescentNIter", 20),
            graph_degree=self.method_param["M"],
            intermediate_graph_degree=self.method_param.get("IntermediateGraphDegree", 128)
        )
        self.index = cagra.build(index_params, X)

    def set_query_arguments(self, ef):
        self.itopk_size = ef
        # Create search parameters (default ef and num_threads can be overridden in self.method_param)
        self.search_params = cagra.SearchParams(
            max_queries=0,
            itopk_size=ef,
            max_iterations=0,
            algo='auto',
            team_size=0,
            search_width=1,
            min_iterations=0,
            thread_block_size=0,
            hashmap_mode='auto',
            hashmap_min_bitlen=0,
            hashmap_max_fill_rate=0.5,
            num_random_samplings=1,
            rand_xor_mask=0x128394,
        )
        self.name = "Cagra  (%s, 'ItopkSize': %s)" % (self.method_param, ef)

    def query(self, v, n):
        # Normalize the vector if the metric is angular
        if self.metric == "inner_product":
            v = cp.asarray(v) / cp.linalg.norm(v)
        if v.dtype != cp.float32:
            v = v.astype(cp.float32)
        
        # Expand dimensions to create a 2D array
        v = v[None, :]  

        # Execute the search and get (distances, neighbors)
        distances, neighbors = cagra.search(self.search_params, self.index, v, n)

        # Return the neighbor indices for the query
        neighbors = cp.asarray(neighbors)
        return neighbors[0].tolist()

    def batch_query(self, X, n):
        # Normalize each vector if the metric is angular
        if self.metric == "inner_product":
            X = cp.asarray(X)
            norms = cp.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
        if X.dtype != cp.float32:
            X = X.astype(cp.float32)
        
        # Execute batch search and store the result in self.res
        self.res = cagra.search(self.search_params, self.index, X, n)

    def get_batch_results(self):
        # Retrieve distances and neighbor indices from the batch search result
        distances, neighbours = self.res
        distances = cp.asarray(distances)
        neighbours = cp.asarray(neighbours)
        results = []
        # Filter out invalid indices (-1) for each query result
        for i in range(len(distances)):
            query_result = []
            for distance, neighbour in zip(distances[i], neighbours[i]):
                if neighbour != -1:
                    query_result.append(int(neighbour))
            results.append(query_result)
        return results

    def __str__(self):
        return "Cagra M=%d nnDescentNIter=%d ItopkSize=%d" % (self.method_param["M"], self.method_param.get("nnDescentNIter", 20), self.itopk_size)
