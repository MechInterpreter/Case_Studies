import pandas as pd
import numpy as np
import json
from scipy.stats import zscore, wasserstein_distance, entropy

class Raw_Data_Handler():
    def __init__(self):
        self.merged_df = None  # Use to run load() function later
        
    def extract(self, customer_information_filename: str, transaction_filename: str, fraud_information_filename: str):
        customer_df = pd.read_csv(customer_information_filename)
        
        with open(fraud_information_filename) as reader:
            fraud_release = json.load(reader)
        fraud_df = pd.json_normalize(fraud_release)  # Flatten JSON file in case of nested structures
        
        transaction_df = pd.read_parquet(transaction_filename)
        
        return customer_df, transaction_df, fraud_df

    def transform(self, customer_information, transaction_information, fraud_information): 
        # Initialize common columns
        common_col_1 = None
        common_col_2 = None
        
        # Detect need for transposition by comparing dimensions and merge fraud_trans conditionally
        if fraud_information.shape[1] == transaction_information.shape[0]:
            print('Transposing fraud_information to align with transaction_information')
            t_fraud_df = fraud_information.T
            # Find common columns to merge on
            common_col_1 = list(set(t_fraud_df.columns).intersection(set(transaction_information.columns)))

            if common_col_1:
                # Merge fraud (transposed) and transaction data via outer join
                fraud_trans_df = pd.merge(t_fraud_df, transaction_information, on=common_col_1, how='outer')
            
        elif fraud_information.shape[0] == transaction_information.shape[1]:
            print('Transposing transaction_information to align with fraud_information')
            t_trans_df = transaction_information.T
            # Find common columns to merge on
            common_col_2 = list(set(t_trans_df.columns).intersection(set(fraud_information.columns)))
            
            if common_col_2:
                # Merge transaction (transposed) and fraud data via outer join
                fraud_trans_df = pd.merge(fraud_information, t_trans_df, on=common_col_2, how='outer')
            
        # Use to detect if automation fails
        common_columns = common_col_1 or common_col_2
        
        # Revert to specific approach for fraud detection if automation fails
        if not common_columns:
            print('No common column names found between fraud and trans data. Reverting to specific approach.')
            
            # Reset index column for renaming columns properly
            t_fraud_df = fraud_information.T.reset_index()
            t_fraud_df.rename(columns={'index': 'trans_num'}, inplace=True)
            t_fraud_df.rename(columns={0: 'fraudulence'}, inplace=True)    
        
            # Merge fraud (transposed) and transaction data via outer join on trans_num column
            fraud_trans_df = pd.merge(t_fraud_df, transaction_information, on='trans_num', how='outer')
        
        common_col_3 = list(set(fraud_trans_df.columns).intersection(set(customer_information.columns)))
        
        if common_col_3:
            # Complete the merge with customer data via outer join on common column
            self.merged_df = pd.merge(customer_information, fraud_trans_df, on=common_col_3, how='outer')
            
            # Complete the merge with customer data via outer join on cc_num column
        else:
            self.merged_df = pd.merge(customer_information, fraud_trans_df, on='cc_num', how='outer')
        
        # Standardize column names with lowercase and underscores for spaces
        self.merged_df.columns = self.merged_df.columns.str.lower().str.replace(' ', '_')
        
        # Drop duplicate and empty rows
        self.merged_df.drop_duplicates(inplace=True)
        self.merged_df.dropna(how='all', inplace=True)
        
        # Drop empty columns
        self.merged_df.dropna(axis=1, how='all', inplace=True)        
        
        return self.merged_df
    
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
        if self.merged_df is None:
            print("No data to save. Run transform() first.")
            
        else:
            try:
                # Save in Parquet format
                self.merged_df.to_parquet(output_filename, index=False)
                print(f"Data successfully saved to {output_filename}")
            except Exception as e:
                print(f"Failed to save data: {e}")