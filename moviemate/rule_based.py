import pandas as pd

class RuleBasedRecommendation:
    def __init__(self, ratings_file, metadata_file):
        self.ratings = pd.read_csv(ratings_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        self.metadata = pd.read_csv(
            metadata_file,
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

    def top_overall(self, top_n=10):
        avg_ratings = self.ratings.groupby('item')['rating'].mean().sort_values(ascending=False)
        top_items = avg_ratings.head(top_n).index
        return self.metadata[self.metadata['item'].isin(top_items)][['item', 'title']]

    def top_by_genre(self, genre, top_n=10):
        genre_items = self.metadata[self.metadata[genre] == 1]['item']
        avg_ratings = self.ratings[self.ratings['item'].isin(genre_items)].groupby('item')['rating'].mean().sort_values(ascending=False)
        top_items = avg_ratings.head(top_n).index
        return self.metadata[self.metadata['item'].isin(top_items)][['item', 'title']]