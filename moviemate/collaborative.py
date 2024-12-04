import pandas as pd
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import Reader
from surprise import SVD, KNNBasic

class CollaborativeFiltering:
    def __init__(self, ratings_file, metadata_file, algorithm='svd', test_size=0.2, random_state=42):
        """
        Initialize the recommender system and load data.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file.
        metadata_file : str
            Path to the item metadata file.
        algorithm : str, optional
            Type of collaborative filtering algorithm ('svd', 'user-user', 'item-item'). Default is 'svd'.
        test_size : float, optional
            Proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            Random seed for reproducibility. Default is 42.
        """
        self.algorithm_type = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.trainset = None
        self.validset = None
        self.model = None
        self.data_file = ratings_file
        self.metadata_file = metadata_file
        self.data = None
        self.items_metadata = None
        self._load_data()

    def _load_data(self):
        """Load the dataset and item metadata, and prepare data for training."""
        df = pd.read_csv(self.data_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        self.data = Dataset.load_from_df(df[['user', 'item', 'rating']], Reader(rating_scale=(1, 5)))

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
            lambda x: ' '.join([col for col in self.items_metadata.columns[6:] if x[col] == 1]),
            axis=1
        )

        # Split data into training and validation sets
        self.trainset, self.validset = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)

    def fit(self):
        """Fit the model on the training set."""
        if self.trainset is None:
            raise ValueError("Data not loaded. Ensure the data file is correctly loaded.")

        # Instantiate the appropriate model based on the algorithm type
        if self.algorithm_type == 'svd':
            self.model = SVD()
        elif self.algorithm_type == 'user-user':
            sim_options = {
                'name': 'cosine',
                'user_based': True
            }
            self.model = KNNBasic(sim_options=sim_options)
        elif self.algorithm_type == 'item-item':
            sim_options = {
                'name': 'cosine',
                'user_based': False
            }
            self.model = KNNBasic(sim_options=sim_options)
        else:
            raise ValueError("Unsupported algorithm type. Use 'svd', 'user-user', or 'item-item'.")

        self.model.fit(self.trainset)

    def evaluate(self):
        """
        Evaluate the model using RMSE on the validation set.

        Returns
        -------
        float
            Root Mean Square Error (RMSE) of the predictions.
        """
        if self.validset is None:
            raise ValueError("Data not loaded. Call `_load_data` first.")
        predictions = self.model.test(self.validset)
        return accuracy.rmse(predictions)

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item.

        Parameters
        ----------
        user_id : int
            ID of the user.
        item_id : int
            ID of the item.

        Returns
        -------
        float
            Predicted rating for the user-item pair.
        """
        return self.model.predict(user_id, item_id).est
    
    def incremental_fit(self, new_data):
        if new_data is None or new_data.empty:
            raise ValueError("New data is empty. Provide valid new interaction data.")

        # Update the ratings DataFrame with new interaction data
        self.ratings_df = pd.concat([self.ratings_df, new_data], ignore_index=True)

        # Reload the data and update trainset
        updated_data = Dataset.load_from_df(self.ratings_df[['user', 'item', 'rating']], Reader(rating_scale=(1, 5)))
        self.trainset = updated_data.build_full_trainset()

        # Fit the model with the updated training set
        self.fit()


if __name__ == "__main__":
    # Performed GridSearch to calculate approximately optimal hyperparameters
    svd_params = {'n_factors': 200, 'n_epochs': 100, 'lr_all': 0.01, 'reg_all': 0.1}
    recommender = CollaborativeFiltering(
        ratings_file='data/u.data',
        metadata_file='data/u.item',
        algorithm=SVD(**svd_params)
    )

    # Fit the model
    recommender.fit()

    # Evaluate the model
    rmse = recommender.evaluate()
    print(f"Validation RMSE: {rmse}")

    # Make a prediction on existing users
    user_id = 196  # Example user
    item_id = 242  # Example movie
    predicted_rating = recommender.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
