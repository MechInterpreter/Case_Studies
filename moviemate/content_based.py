import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np


class ContentBasedFiltering:
    """
    A content-based recommender system.

    This class uses item metadata and user ratings to recommend items
    based on their similarity to items the user has interacted with.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the content-based recommender system.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file (user, item, rating).
        metadata_file : str
            Path to the item metadata file (item, features).
        """
        self.ratings_file = ratings_file
        self.metadata_file = metadata_file
        self.item_profiles = None
        self.user_profiles = None
        self.similarity_matrix = None
        self.ratings = None
        self.items_metadata = None
        self._load_data()
        self._build_item_profiles()

    def _load_data(self):
        """Load the ratings and item metadata datasets."""
        self.ratings = pd.read_csv(
            self.ratings_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        self.items_metadata = pd.read_csv(
            self.metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )

        # Combine genre columns into a single 'features' column
        self.items_metadata['features'] = self.items_metadata.iloc[:, 6:].apply(
            lambda x: ' '.join([
                col for col in self.items_metadata.columns[6:] if x[col] == 1
            ]),
            axis=1
        )

    def _build_item_profiles(self):
        """Create item profiles based on item features using TF-IDF."""
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.items_metadata['features'])
        self.item_profiles = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=self.items_metadata['item']
        )
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _get_user_profile(self, user_id):
        """Create a user profile based on the user's past ratings."""
        user_ratings = self.ratings[self.ratings['user'] == user_id]
        if user_ratings.empty:
            return np.zeros(self.item_profiles.shape[1])

        user_profile = np.zeros(self.item_profiles.shape[1])

        for _, row in user_ratings.iterrows():
            item_vector = self.item_profiles.loc[row['item']].values
            user_profile += item_vector * row['rating']

        # Normalize by number of items rated to avoid overemphasizing heavy raters
        if len(user_ratings) > 0:
            user_profile /= len(user_ratings)

        # Normalize the user profile vector
        return user_profile / np.linalg.norm(user_profile) if np.linalg.norm(user_profile) != 0 else user_profile

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item using content-based similarity.

        Parameters
        ----------
        user_id : int
            ID of the user.
        item_id : int
            ID of the item.

        Returns
        -------
        float
            Predicted rating.
        """
        # Handle new users by providing a default rating strategy
        if user_id not in self.ratings['user'].unique():
            # Predict based on item genre (if available) or average rating
            if item_id in self.items_metadata['item'].unique():
                # Get genres for the item
                item_genres = self.items_metadata[self.items_metadata['item'] == item_id]['features'].iloc[0].split()
                if item_genres:
                    # Calculate average rating for each genre and return the average
                    genre_ratings = []
                    for genre in item_genres:
                        genre_items = self.items_metadata[self.items_metadata['features'].str.contains(genre)]['item']
                        avg_rating = self.ratings[self.ratings['item'].isin(genre_items)]['rating'].mean()
                        if not np.isnan(avg_rating):
                            genre_ratings.append(avg_rating)
                    if genre_ratings:
                        return np.mean(genre_ratings)
                # If genre-based ratings not available, fall back to overall average
                avg_rating = self.ratings['rating'].mean()
                return avg_rating if not np.isnan(avg_rating) else 3.0
            else:
                return 3.0  # Default rating for a completely new item

        # Handle new items that are not in the item profiles
        if item_id not in self.item_profiles.index:
            # Predict using a default rating as there's no item profile available
            avg_rating = self.ratings['rating'].mean()
            return avg_rating if not np.isnan(avg_rating) else 3.0  # Default rating

        # If both user and item exist, proceed with standard content-based prediction
        user_profile = self._get_user_profile(user_id)
        item_vector = self.item_profiles.loc[item_id].values

        # Compute similarity-based rating
        if np.linalg.norm(user_profile) == 0 or np.linalg.norm(item_vector) == 0:
            # If either profile has no meaningful data, fall back to a default rating
            return 3.0

        return np.dot(user_profile, item_vector) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_vector)
        )

    def evaluate(self, sample_size=1000):
        """
        Evaluate the model by calculating the RMSE on a sample of user-item ratings.

        Parameters
        ----------
        sample_size : int, optional
            Number of random user-item pairs to evaluate. Default is 1000.

        Returns
        -------
        float
            RMSE value.
        """
        sample_ratings = self.ratings.sample(n=sample_size, random_state=42)

        true_ratings = []
        predicted_ratings = []

        for _, row in sample_ratings.iterrows():
            user_id, item_id, true_rating = row['user'], row['item'], row['rating']
            try:
                predicted_rating = 5 * self.predict(user_id, item_id)
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating)
            except ValueError:
                continue

        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    
    def get_top_items(self, user_id, top_n=5):
        """
        Get the top-n items for a user based on their user profile.

        Parameters
        ----------
        user_id : int
            ID of the user for whom we want recommendations.
        top_n : int
            Number of top items to return.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the top-n recommended items and their scores.
        """
        if user_id not in self.ratings['user'].unique():
            # Cold start scenario, recommend top-n popular items
            top_items = (
                self.ratings.groupby('item')['rating']
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
                .rename(columns={"item": "item_id", "rating": "score"})
            )
            return top_items

        # Calculate user profile
        user_profile = self._get_user_profile(user_id)
        scores = []

        for item_id in self.item_profiles.index:
            item_vector = self.item_profiles.loc[item_id].values
            # Calculate cosine similarity
            score = np.dot(user_profile, item_vector) / (
                np.linalg.norm(user_profile) * np.linalg.norm(item_vector)
            )
            scores.append((item_id, score))

        # Sort by similarity scores in descending order and return top-n items
        top_items = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        return top_items


if __name__ == "__main__":
    recommender = ContentBasedFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )

    rmse = recommender.evaluate(sample_size=100)
    print(f"RMSE on sample: {rmse}")

    user_id = 1  # Example user
    item_id = 242  # Example movie
    predicted_rating = recommender.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")