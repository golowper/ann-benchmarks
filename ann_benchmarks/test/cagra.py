import cupy as cp
from cuvs.neighbors import cagra, hnsw
import time

n_samples = 1000000
n_features = 100
n_queries = 1000
M = 64
dataset = cp.random.random_sample((n_samples, n_features),
                                  dtype=cp.float32)
# Build index
index_params = cagra.IndexParams(
    metric="inner_product", 
    build_algo='nn_descent',
    graph_degree=M
    )
start = time.time()
index = cagra.build(index_params, dataset)
end = time.time()
print("Time taken to build index: ", end-start)

# Search using the built index
queries = cp.random.random_sample((n_queries, n_features),
                                  dtype=cp.float32)
k = 10

search_params = cagra.SearchParams(
    itopk_size=500
)
# Execute the search and get (distances, neighbors)
start = time.time()
distances, neighbors = cagra.search(search_params, index, queries,k)
end = time.time()
print("Time taken to search: ", end-start)
distances = cp.asarray(distances)
neighbors = cp.asarray(neighbors)

# search_params = hnsw.SearchParams(
#     ef=200,
#     num_threads=0
# )
# # Convert CAGRA index to HNSW
# hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)
# # Using a pooling allocator reduces overhead of temporary array
# # creation during search. This is useful if multiple searches
# # are performed with same query size.
# queries_host = cp.asnumpy(queries)
# distances, neighbors = hnsw.search(search_params, hnsw_index, queries_host, k)

# print min distance and the corresponding neighbor index

# print(distances.shape, neighbors.shape)

# for i in range(10):
#     print(distances[i].min(), neighbors[i][distances[i].argmin()])
