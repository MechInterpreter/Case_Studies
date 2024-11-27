import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collaborative import CollaborativeFiltering
from content_based import ContentBasedFiltering
from diversifier import Diversifier
from recommender import Recommender

class Pipeline:
    def __init__(self):
        self.ratings_df = None
        self.collaborative_model = None
        self.content_based_model = None
        self.diversifier = None
        self.recommender = None

    def load_dataset(self, file_path):
        try:
            column_names = ['user_id', 'item_id', 'rating', 'timestamp']
            self.ratings_df = pd.read_csv(file_path, sep='\t', names=column_names, encoding='latin-1')
            print(f"Dataset loaded with shape: {self.ratings_df.shape}")
            print(self.ratings_df.head())
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")

    def partition_data(self, partition_type=None, test_size=0.2, random_state=42):
        if self.ratings_df is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        if partition_type == 'user_stratified':
            # User-stratified split: Split users into train and test sets
            unique_users = self.ratings_df['user_id'].unique()
            train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)
            train_df = self.ratings_df[self.ratings_df['user_id'].isin(train_users)]
            test_df = self.ratings_df[self.ratings_df['user_id'].isin(test_users)]
        elif partition_type == 'temporal':
            # Temporal split: Sort by timestamp and then split
            self.ratings_df = self.ratings_df.sort_values(by='timestamp')
            split_index = int(len(self.ratings_df) * (1 - test_size))
            train_df = self.ratings_df.iloc[:split_index]
            test_df = self.ratings_df.iloc[split_index:]
        else:
            # Default random split (similar to train_test_split)
            train_df, test_df = train_test_split(self.ratings_df, test_size=test_size, random_state=random_state)

        print(f"Training set shape: {train_df.shape}")
        print(f"Testing set shape: {test_df.shape}")
        return train_df, test_df

    def initialize_models(self, ratings_file, metadata_file):
        # Initialize Collaborative Filtering model
        self.collaborative_model = CollaborativeFiltering(ratings_file=ratings_file, metadata_file=metadata_file)
        
        # Initialize Content-Based Filtering model
        self.content_based_model = ContentBasedFiltering(ratings_file=ratings_file, metadata_file=metadata_file)
        
        # Initialize Diversifier
        self.diversifier = Diversifier(metadata=self.content_based_model.items_metadata)
        
        # Initialize Hybrid Recommender
        self.recommender = Recommender(model=self.collaborative_model)

    def fit_models(self):
        if self.collaborative_model:
            self.collaborative_model.fit()
        else:
            print("Collaborative model not initialized.")

    def evaluate_models(self):
        if self.collaborative_model:
            rmse_collaborative = self.collaborative_model.evaluate()
            print(f"Collaborative Filtering Model RMSE: {rmse_collaborative}")
        if self.content_based_model:
            rmse_content_based = self.content_based_model.evaluate(sample_size=100)
            print(f"Content-Based Filtering Model RMSE: {rmse_content_based}")

    def get_recommendations(self, user_id, top_n=10):
        if self.recommender:
            rankings = self.recommender.rank_items(user_id=user_id, top_n=top_n)
            if self.diversifier:
                reranked = self.diversifier.rerank(rankings, top_n=top_n)
                print(reranked)
            else:
                print(rankings)
        else:
            print("Recommender not initialized.")