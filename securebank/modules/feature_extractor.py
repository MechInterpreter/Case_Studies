import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import zscore, wasserstein_distance, entropy

class Feature_Extractor():
    def __init__(self, target_column_name):
        self.target_column_name = target_column_name
            
    def extract(self, training_dataset_filename: str, testing_dataset_filename: str):
        training_df = pd.DataFrame(training_dataset_filename)
        testing_df = pd.DataFrame(testing_dataset_filename)
        
        return training_df, testing_df

    def transform(self, training_dataset, testing_dataset):
        # Separate features and target
        X_train = training_dataset.drop(columns=[self.target_column_name])
        Y_train = training_dataset[self.target_column_name]
        
        X_test = testing_dataset.drop(columns=[self.target_column_name])
        Y_test = testing_dataset[self.target_column_name]
        
        # Only select numerical features for correlation purposes
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        # Define transformation: scaling for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # Standardize numerical features
        ])
        
        # Apply scaling transformation only to numerical features
        X_train_transformed = numerical_transformer.fit_transform(X_train[numerical_features])
        X_test_transformed = numerical_transformer.transform(X_test[numerical_features])
        
        # Convert back to DataFrame with appropriate numerical feature columns
        train_df_transformed = pd.DataFrame(X_train_transformed, columns=numerical_features)
        test_df_transformed = pd.DataFrame(X_test_transformed, columns=numerical_features)
        partitioned_data = [train_df_transformed, test_df_transformed]
        
        return partitioned_data
    
    # Define function that computes quality metrics
    def describe(self, partitioned_data, **kwargs):
        descriptions = []  # Store descriptions for both DataFrames
        
        for df in partitioned_data:
            description = {}
            description['description'] = {}
            
            # Correlation Matrix
            try:
                description['description']['correlation'] = df.corr()  # Pearson by default
            except Exception as e:
                description['description']['correlation'] = f"Error calculating correlation: {str(e)}"
            
            # Completeness
            description['description']['table_completeness'] = df.notnull().sum().sum() / (df.shape[0] * df.shape[1])
            description['description']['record_completeness'] = df.dropna().shape[0] / df.shape[0]   
            description['description']['column_completeness'] = (df.notnull().sum() / df.shape[0]).to_dict()
            
            # Size
            description['description']['num_samples'] = df.shape[0]
            description['description']['num_features'] = df.shape[1]
        
            # Accuracy
            try:
                z_scores = zscore(df.select_dtypes(include=[float, int]))
                
                # Convert z-scores back into a DataFrame to use .to_dict()
                z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=[float, int]).columns)

                # Identify outliers
                outliers = ((z_scores_df < -3) | (z_scores_df > 3)).sum(axis=0)
                
                # Convert outliers to a dictionary
                description['description']['z_score_outliers'] = outliers.to_dict()
                
            except Exception as e:
                description['description']['z_score_outliers'] = f"Error calculating z-scores: {str(e)}"

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
        
            # Append description for this DataFrame to the list
            descriptions.append(final_description)
    
        return descriptions  # Return the list of descriptions for all DataFrames