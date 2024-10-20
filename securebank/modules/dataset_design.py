import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import zscore, wasserstein_distance, entropy

class Dataset_Designer():
    def __init__(self, test_size: float, target_column_name: str):
        self.df = None  # Use to run load() function later
        self.test_size = test_size  # Use to partition data later
        self.target_column_name = target_column_name
        
    def extract(self, raw_dataset_filename: str):
        transaction_df = pd.read_parquet(raw_dataset_filename)
        
        return transaction_df

    def sample(self, raw_dataset): 
        X = raw_dataset.drop(columns=[self.target_column_name])  # Features
        Y = raw_dataset[self.target_column_name]  # Target column
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        
        # Add column to differentiate between train and test sets
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['set_type'] = 'train'
        X_test['set_type'] = 'test'
        
        # Combine X and Y for both training and testing sets
        train_df = pd.concat([X_train, Y_train], axis=1)
        test_df = pd.concat([X_test, Y_test], axis=1)
        
        # Combine train and test DataFrames into one DataFrame
        self.df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        
        return self.df
    
    # Define function that computes quality metrics
    def describe(self, df, **kwargs):
        description = {}
        description['description'] = {}
        
        # Completeness
        description['description']['table_completeness'] = df.notnull().sum().sum() / (df.shape[0] * df.shape[1])
        description['description']['record_completeness'] = df.dropna().shape[0] / df.shape[0]   
        description['description']['column_completeness'] = (df.notnull().sum() / df.shape[0]).to_dict()
        
        # Size
        description['description']['num_samples'] = df.shape[0]
        description['description']['num_features'] = df.shape[1]
    
        # Accuracy
        z_scores = zscore(df.select_dtypes(include=[float, int]))
        outliers = ((z_scores < -3) | (z_scores > 3)).sum(axis=0)
        description['description']['z_score_outliers'] = outliers.to_dict()
        
        # Timeliness
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            time_diff = df['trans_date_trans_time'].diff().dt.total_seconds()
            mean_time_diff = time_diff.mean()
            variance_time_diff = time_diff.var()
            description['description']['mean_time_diff'] = mean_time_diff
            description['description']['variance_time_diff'] = variance_time_diff
            
            description['description']['update_frequency'] = df['trans_date_trans_time'].diff().mean()
        
        # Representativeness
        if 'fraudulence' in df.columns:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[float, int])

            # Split based on 'fraudulence' values 0.0 and 1.0
            dist_1 = numeric_df[df['fraudulence'] == 0.0].values.flatten()
            dist_2 = numeric_df[df['fraudulence'] == 1.0].values.flatten()

            # Remove NaN values from both distributions
            dist_1 = dist_1[~np.isnan(dist_1)]
            dist_2 = dist_2[~np.isnan(dist_2)]
            
            # Ensure both distributions have the same length by padding the shorter one with zeros
            max_len = max(len(dist_1), len(dist_2))
            dist_1 = np.pad(dist_1, (0, max_len - len(dist_1)), 'constant')
            dist_2 = np.pad(dist_2, (0, max_len - len(dist_2)), 'constant')

            # Min-max scaling
            dist_1_scaled = (dist_1 - dist_1.min()) / (dist_1.max() - dist_1.min())
            dist_2_scaled = (dist_2 - dist_2.min()) / (dist_2.max() - dist_2.min())

            # Check if both distributions have data
            if len(dist_1) > 0 and len(dist_2) > 0:
                # Calculate Wasserstein distance
                description['description']['wasserstein_distance'] = wasserstein_distance(dist_1, dist_2)

                # Avoid zero issues in KL divergence by adding epsilon
                epsilon = 1e-10
                dist_1_prob = (dist_1_scaled + epsilon) / (dist_1_scaled.sum() + epsilon * len(dist_1_scaled))
                dist_2_prob = (dist_2_scaled + epsilon) / (dist_2_scaled.sum() + epsilon * len(dist_2_scaled))

                # Ensure both distributions are valid (non-empty, no zero sums)
                if dist_1.sum() > 0 and dist_2.sum() > 0:
                    description['description']['kl_divergence'] = entropy(dist_1_prob, dist_2_prob)
                else:
                    description['description']['kl_divergence'] = 'Undefined (zero sum in distribution)'
        
        # Build final description dictionary with version and storage path
        final_description = {
            'version': kwargs.get('version_name', '1.0'),
            'storage': kwargs.get('storage_path', 'unknown'),
            'description': description['description']
        }
    
        return final_description
    
    # Define load function to save merged_df in Parquet format
    def load(self, output_filename: str):
        if self.df is None:
            print("No data to save. Run sample() first.")
            
        else:
            try:
                # Save in Parquet format
                self.df.to_parquet(output_filename, index=False)
                print(f"Data successfully saved to {output_filename}")
            except Exception as e:
                print(f"Failed to save data: {e}")