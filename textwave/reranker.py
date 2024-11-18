from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

class Reranker:
    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the Reranker class with specified reranking type and model name.

        :param type: A string indicating the type of reranking ('cross_encoder', 'tfidf', or 'hybrid').
        :param cross_encoder_model_name: A string specifying the cross-encoder model name (default is 'cross-encoder/ms-marco-MiniLM-L-6-v2').
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)

    def rerank(self, query, context, distance_metric="cosine"):
        """
        Selects the reranking method based on the initialized type.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document to be reranked.
        :param distance_metric: A string indicating the distance metric to use for TF-IDF reranking (default is "cosine").
        :return: Ranked documents, indices, and scores based on the selected reranking method.
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)

    def cross_encoder_rerank(self, query, context):
        """
        Reranks documents based on relevance to the query using a cross-encoder model.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :return: A list of ranked documents, their indices, and relevance scores.
        """
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            relevance_scores = logits.squeeze().tolist()

        ranked_indices = torch.argsort(torch.tensor(relevance_scores), descending=True).tolist()
        ranked_documents = [context[idx] for idx in ranked_indices]
        scores = [relevance_scores[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Reranks documents based on their similarity to the query using TF-IDF and a specified distance metric.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :param distance_metric: The distance metric to use for similarity calculation ('cosine', 'euclidean', 'manhattan', etc.).
        :return: A list of ranked documents, their indices, and similarity scores.
        """
        all_texts = [query] + context
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate distance scores between the query (first vector) and each document
        distances = pairwise_distances(tfidf_matrix[0:1], tfidf_matrix[1:], metric=distance_metric).flatten()
        
        # Sort indices in ascending order for similarity metrics (higher is better) and descending for distance
        if distance_metric == "cosine":
            ranked_indices = np.argsort(distances)
        else:
            ranked_indices = np.argsort(distances)[::-1]

        ranked_documents = [context[idx] for idx in ranked_indices]
        scores = [distances[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combines TF-IDF and cross-encoder scores for hybrid reranking.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :param tfidf_weight: Weight for the TF-IDF score in the combined ranking.
        :param cross_encoder_weight: Weight for the cross-encoder score in the combined ranking.
        :return: A list of ranked documents, indices, and combined scores.
        """
        tfidf_documents, tfidf_indices, tfidf_scores = self.tfidf_rerank(query, context, distance_metric)
        cross_encoder_docs, _, cross_encoder_scores = self.cross_encoder_rerank(query, tfidf_documents)

        combined_scores = []
        for i, doc in enumerate(cross_encoder_docs):
            tfidf_score = tfidf_scores[tfidf_indices[i]]
            cross_encoder_score = cross_encoder_scores[i]
            combined_score = tfidf_weight * tfidf_score + (1-tfidf_weight) * cross_encoder_score
            combined_scores.append((doc, tfidf_indices[i], combined_score))

        combined_scores = sorted(combined_scores, key=lambda x: x[2], reverse=True)

        ranked_documents = [doc for doc, _, _ in combined_scores]
        ranked_indices = [idx for _, idx, _ in combined_scores]
        scores = [score for _, _, score in combined_scores]

        return ranked_documents, ranked_indices, scores

    def sequential_rerank(self, query, context, **kwargs):
        top_k = kwargs.get('top_k', 5)  # Default to 5 if not provided

        # Apply TF-IDF reranking
        tfidf_documents, tfidf_indices, tfidf_scores = self.tfidf_rerank(query, context)

        # Select top_k documents from TF-IDF results to rerank with cross-encoder
        top_docs = tfidf_documents[:top_k]

        # Apply cross-encoder reranking on top TF-IDF documents
        cross_encoder_docs, cross_encoder_indices, cross_encoder_scores = self.cross_encoder_rerank(query, top_docs)

        # Map back indices to original context
        final_indices = [tfidf_indices[idx] for idx in cross_encoder_indices]
        final_documents = [context[idx] for idx in final_indices]
        final_scores = cross_encoder_scores

        return final_documents, final_indices, final_scores

    def tfidf_corpus_rerank(self, query, corpus):
        all_texts = [query] + corpus
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_documents = [corpus[idx] for idx in ranked_indices]
        scores = [similarities[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores