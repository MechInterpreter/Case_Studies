import numpy as np

class FaissSearch:
    def __init__(self, faiss_index, metric='euclidean', p=3):
        """
        Initialize the search class with a FaissIndex instance and distance metric.
        
        :param faiss_index: A FaissIndex instance.
        :param metric: The distance metric ('euclidean', 'dot_product', 'cosine', 'minkowski').
        :param p: The parameter for Minkowski distance (p=2 for Euclidean, p=1 for Manhattan).
        """
        self.index = faiss_index.index
        self.metric = metric
        self.p = p  # Minkowski distance parameter
        self.faiss_index = faiss_index

    def search(self, query_vector, k=5):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        
        :param query_vector: The vector to query (numpy array).
        :param k: Number of nearest neighbors to return.
        :return: Distances, indices, and metadata of the nearest neighbors.
        """
        if self.metric == 'cosine':
            # Normalize vectors for cosine similarity
            query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
            distances, indices = self.index.search(query_vector, k)
            # Convert dot product to cosine distance: distance = 1 - cosine similarity
            distances = 1 - distances

        elif self.metric == 'minkowski':
            # Use FAISS to perform an approximate Euclidean search
            distances, indices = self.index.search(query_vector, k)
            # Manually compute Minkowski distance based on the returned vectors
            nearest_vectors = [self.index.reconstruct(int(i)) for i in indices[0]]
            distances = self._compute_minkowski(query_vector[0], nearest_vectors, p=self.p)

        else:
            # Default FAISS search (Euclidean or Dot Product)
            distances, indices = self.index.search(query_vector, k)

        # Retrieve metadata for the nearest neighbors
        metadata_results = [self.faiss_index.get_metadata(int(i)) for i in indices[0]]
        return distances, indices, metadata_results

    def _compute_minkowski(self, query_vector, nearest_vectors, p):
        """
        Compute Minkowski distance between the query vector and the nearest neighbors.
        
        :param query_vector: The query vector (numpy array).
        :param nearest_vectors: List of nearest neighbor vectors (numpy arrays).
        :param p: Parameter for Minkowski distance (p=2 is Euclidean, p=1 is Manhattan).
        :return: List of Minkowski distances.
        """
        distances = []
        for vec in nearest_vectors:
            distance = np.sum(np.abs(query_vector - vec) ** p) ** (1 / p)
            distances.append(distance)
        return distances