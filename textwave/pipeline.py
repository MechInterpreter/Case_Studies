from embedding import Embedding
from preprocessing import DocumentProcessing
from question_answering import QA_Generator
from indexing import FaissIndex
from reranker import Reranker
from search import FaissSearch
import os

class Pipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', reranker_type='hybrid'):
        self.processor = DocumentProcessing()
        self.embedder = Embedding(embedding_model)
        self.index = FaissIndex(index_type='brute_force')
        self.qa_generator = QA_Generator()
        self.context_data = []
        self.reranker = Reranker(type=reranker_type)
    
    def preprocess_corpus(self, corpus_directory, chunking_strategy='sentence', fixed_length=None, overlap_size=2):
        for filename in os.listdir(corpus_directory):
            file_path = os.path.join(corpus_directory, filename)

            # Chunking strategy
            if chunking_strategy == 'sentence':
                chunks = self.processor.sentence_chunking(file_path, overlap_size)
            elif chunking_strategy == 'fixed-length' and fixed_length is not None:
                chunks = self.processor.fixed_length_chunking(file_path, fixed_length, overlap_size)
            else:
                raise ValueError("Invalid chunking strategy or missing fixed_length for 'fixed-length' strategy.")

            # Store chunks for later inspection
            self.context_data.extend(chunks)

            # Debug output for chunks
            print(f"Generated {len(chunks)} chunks from '{filename}':")
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx + 1}: '{chunk}'")

            # Continue with encoding and indexing
            if not chunks:
                continue

            # Embed and index each chunk
            encoded_chunks = self.embedder.encode(chunks)
            if encoded_chunks is not None and len(encoded_chunks) > 0:
                self.vector_dimension = encoded_chunks.shape[1]
                metadata = [{"document": filename, "chunk": i, "text": chunk} for i, chunk in enumerate(chunks)]
                self.index.add_embeddings(encoded_chunks, metadata=metadata)
                print(f"Added {len(encoded_chunks)} embeddings for '{filename}' to the FAISS index.")
            else:
                print(f"No embeddings generated for '{filename}'.")

        # Debug statement for FAISS index population
        if hasattr(self.index, 'index'):
            print(f"FAISS index has {self.index.index.ntotal} items.")

    def save_index(self, faiss_path="faiss.index", metadata_path="metadata.pkl"):
        self.index.save(faiss_path, metadata_path)
        
    def load_index(self, faiss_path="faiss.index", metadata_path="metadata.pkl"):
        self.index = FaissIndex()
        try:
            self.index.load(faiss_path, metadata_path)
            if not hasattr(self.index, 'index') or self.index.index is None:
                print("Failed to load FAISS index. Index is None.")
            else:
                print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {str(e)}")

    def __encode(self, query):
        return self.embedder.encode([query])

    def search_neighbors(self, query_embedding, k=10):
        if not hasattr(self.index, 'index') or self.index.index is None:
            print("No FAISS index found. Cannot perform search.")
            return []
        
        faiss_search = FaissSearch(self.index, metric="euclidean")
        distances, indices, metadata = faiss_search.search(query_embedding, k)

        if not metadata:
            print("No neighbors found for the query.")
        else:
            print(f"Found {len(metadata)} neighbors.")
            for i, meta in enumerate(metadata):
                print(f"Neighbor {i + 1}: {meta.get('text', 'No text available')}")

        neighbors = [meta["text"] for meta in metadata if meta]
        return neighbors

    def generate_answer(self, query, context, rerank=True, rerank_top_k=5):
        if not context:
            print("No context found for the query.")
            return "No context"

        # Apply reranking if specified and there is more than one context to rerank
        if rerank and len(context) > 1:
            context, _, _ = self.reranker.rerank(query, context)

            # Filter top k after reranking
            context = context[:rerank_top_k]

        # Print the context to check if it's relevant
        print(f"Top {rerank_top_k} context(s) used for answering '{query}':")
        for idx, ctx in enumerate(context):
            print(f"Context {idx + 1}: {ctx}")

        return self.qa_generator.generate_answer(query, context)

    def query(self, query, k=5, rerank=True, rerank_top_k=None):
        # Encode the query
        query_embedding = self.__encode(query)

        # Search for nearest neighbors
        neighbors = self.search_neighbors(query_embedding, k)

        # Apply reranking if specified
        if rerank and len(neighbors) > 1:
            neighbors, _, _ = self.reranker.rerank(query, neighbors)

            # If rerank_top_k is specified, filter the top-k reranked contexts
            if rerank_top_k:
                neighbors = neighbors[:rerank_top_k]

        # Generate the answer using the retrieved and optionally reranked context
        return self.generate_answer(query, neighbors)