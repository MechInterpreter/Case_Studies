from embedding import Embedding
from preprocessing import DocumentProcessing
from question_answering import QA_Generator
from indexing import FaissIndex
from reranker import Reranker
from search import FaissSearch
import os

class Pipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.processor = DocumentProcessing()
        self.embedder = Embedding(embedding_model)
        self.index = FaissIndex(index_type='brute_force')  # Initialize FAISS index here
    
    def preprocess_corpus(self, corpus_directory, chunking_strategy='sentence', fixed_length=None, overlap_size=2):
        for filename in os.listdir(corpus_directory):
            file_path = os.path.join(corpus_directory, filename)
            if chunking_strategy == 'sentence':
                chunks = self.processor.sentence_chunking(file_path, overlap_size)
            elif chunking_strategy == 'fixed-length' and fixed_length is not None:
                chunks = self.processor.fixed_length_chunking(file_path, fixed_length, overlap_size)
            else:
                raise ValueError("Invalid chunking strategy or missing fixed_length for 'fixed-length' strategy.")
            
            # Store chunks for later inspection
            self.chunks.extend(chunks)

            # After generating chunks in preprocess_corpus
            print(f"Generated {len(chunks)} chunks:")
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx+1}: '{chunk}'")

            # Continue with encoding and indexing
            if not chunks:
                continue
            
            # Embed and index each chunk
            encoded_chunks = self.embedder.encode(chunks)
            metadata = [{"document": filename, "chunk": i, "text": chunk} for i, chunk in enumerate(chunks)]
            self.index.add_embeddings(encoded_chunks, metadata=metadata)
            
    def save_index(self, faiss_path="faiss.index", metadata_path="metadata.pkl"):
        self.index.save(faiss_path, metadata_path)
        
    def load_index(self, faiss_path="faiss.index", metadata_path="metadata.pkl"):
        self.index = FaissIndex()
        self.index.load(faiss_path, metadata_path)

    def search(self, query, k=5, rerank=False, reranker_type="hybrid"):
        query_vector = self.embedder.encode([query])
        faiss_search = FaissSearch(self.index, metric="euclidean")
        distances, indices, metadata = faiss_search.search(query_vector, k)
        
        if rerank:
            reranker = Reranker(type=reranker_type)
            ranked_documents, _, scores = reranker.rerank(query, [meta["document"] for meta in metadata])
            return ranked_documents, scores
        else:
            return metadata, distances