import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class Feature_Extractor():
    def __init__(self, target_column_name, exclude_features=None):
        self.target_column_name = target_column_name
        self.frequency_encoding_maps = {}  # Initialize a dictionary to store frequency encoding maps
        
        # Initialize imputers and scaler
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        
        self.numerical_cols = None
        self.categorical_cols = None
        self.high_cardinality = ['merchant']
        
        # Set default exclude_features if none provided
        if exclude_features is None:
            # Initial exclude features
            self.exclude_features = ['index', 'first', 'last', 'street', 'trans_num', 
                                     'sex', 'city', 'state', 'zip', 'lat', 'long', 
                                     'city_pop', 'job', 'category', 'age']
        else:
            self.exclude_features = exclude_features
        
    def extract(self, partitioned_dataset_filename: str):
        # Read the partitioned dataset
        partitioned_df = pd.read_parquet(partitioned_dataset_filename)
        
        # Split into training and testing DataFrames based on 'set_type' column
        train_df = partitioned_df[partitioned_df['set_type'] == 'train'].drop(columns=['set_type'])
        test_df = partitioned_df[partitioned_df['set_type'] == 'test'].drop(columns=['set_type'])
        
        return train_df, test_df
    
    def fit_transformers(self, X_train):
        # Identify numerical and categorical columns
        self.numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

        # Remove target column if present
        if self.target_column_name in self.numerical_cols:
            self.numerical_cols.remove(self.target_column_name)
        if self.target_column_name in self.categorical_cols:
            self.categorical_cols.remove(self.target_column_name)
        
        # Fit numerical and categorical imputers, and scaler on training data
        if self.numerical_cols:
            self.numerical_imputer.fit(X_train[self.numerical_cols])
            self.scaler.fit(X_train[self.numerical_cols])
            
            # Save the scaler and imputer
            joblib.dump(self.scaler, 'scaler.joblib')
            joblib.dump(self.numerical_imputer, 'numerical_imputer.joblib')
        
        if self.categorical_cols:
            self.categorical_imputer.fit(X_train[self.categorical_cols])

    def transform_data(self, X):
        # Impute numerical features
        if self.numerical_cols:
            X[self.numerical_cols] = self.numerical_imputer.transform(X[self.numerical_cols])
        
        # Impute categorical features
        if self.categorical_cols:
            X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])

        # Handling NaN values for high-cardinality categorical features
        for col in self.categorical_cols:
            if col in self.high_cardinality:
                X[col] = X[col].fillna('Unknown')  # Replace NaN with a placeholder

        # Scale numerical features
        if self.numerical_cols:
            X[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])

        return X

    
    def time_interval(self, df):
        # Ensure 'trans_date_trans_time' is datetime
        if df['trans_date_trans_time'].dtype != 'datetime64[ns]':
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        
        # Sort by customer ID and transaction time
        df = df.sort_values(['cc_num', 'trans_date_trans_time'])
        
        # Compute time difference in minutes
        df['time_since_last_trans'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 60
        
        # Fill NaN values (first transaction for each customer) with a default value
        df['time_since_last_trans'] = df['time_since_last_trans'].fillna(df['time_since_last_trans'].max())
        
        return df

    def transform(self, training_dataset, testing_dataset):
        # Drop rows with NaN in target column
        print("Number of NaN values in target before processing:")
        print("Training:", training_dataset[self.target_column_name].isna().sum())
        print("Testing:", testing_dataset[self.target_column_name].isna().sum())
        
        training_dataset = training_dataset.dropna(subset=[self.target_column_name])
        testing_dataset = testing_dataset.dropna(subset=[self.target_column_name])
        
        print("\nNumber of NaN values in target after dropping:")
        print("Training:", training_dataset[self.target_column_name].isna().sum())
        print("Testing:", testing_dataset[self.target_column_name].isna().sum())
        
        # Before transformation
        train_cc_nums = set(training_dataset['cc_num'].unique())
        test_cc_nums = set(testing_dataset['cc_num'].unique())
        common_cc_nums = train_cc_nums.intersection(test_cc_nums)
        
        if common_cc_nums:
            print(f"\nOverlap detected in cc_num values between training and testing sets: {len(common_cc_nums)} overlaps")
        else:
            print("\nNo overlap in cc_num values between training and testing sets.")

        # Separate features and target
        X_train = training_dataset.drop(columns=[self.target_column_name]).copy()
        Y_train = training_dataset[self.target_column_name].copy()

        X_test = testing_dataset.drop(columns=[self.target_column_name]).copy()
        Y_test = testing_dataset[self.target_column_name].copy()

        # Exclude irrelevant features using self.exclude_features
        X_train = X_train.drop(columns=self.exclude_features, errors='ignore')
        X_test = X_test.drop(columns=self.exclude_features, errors='ignore')

        # Convert date columns to datetime
        X_train['trans_date_trans_time'] = pd.to_datetime(X_train['trans_date_trans_time'], errors='coerce')
        X_test['trans_date_trans_time'] = pd.to_datetime(X_test['trans_date_trans_time'], errors='coerce')

        # Feature Engineering
        X_train['transaction_hour'] = X_train['trans_date_trans_time'].dt.hour
        X_test['transaction_hour'] = X_test['trans_date_trans_time'].dt.hour

        # Compute time since last transaction
        X_train = self.time_interval(X_train)
        X_test = self.time_interval(X_test)

        # Drop original date column and 'cc_num'
        X_train = X_train.drop(columns=['trans_date_trans_time', 'cc_num', 'dob'], errors='ignore')
        X_test = X_test.drop(columns=['trans_date_trans_time', 'cc_num', 'dob'], errors='ignore')

        # Update numerical and categorical columns after feature engineering
        self.numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

        # Remove target column if present
        if self.target_column_name in self.numerical_cols:
            self.numerical_cols.remove(self.target_column_name)
        if self.target_column_name in self.categorical_cols:
            self.categorical_cols.remove(self.target_column_name)

        # Fit transformers on training data
        self.fit_transformers(X_train)

        # Transform training and testing data
        X_train = self.transform_data(X_train)
        X_test = self.transform_data(X_test)

        # Encode Categorical Variables
        categorical_features = ['merchant', 'transaction_day_of_week']
        categorical_features = [col for col in categorical_features if col in X_train.columns]

        # Handle 'sex' with label encoding
        if 'sex' in categorical_features:
            le = LabelEncoder()
            X_train['sex'] = le.fit_transform(X_train['sex'])
            X_test['sex'] = le.transform(X_test['sex'])
            categorical_features.remove('sex')

        # Frequency encoding for high cardinality features
        high_cardinality_features = ['merchant', 'category']
        high_cardinality_features = [col for col in high_cardinality_features if col in X_train.columns]

        def frequency_encoding(column, df_train, df_test):
            freq_encoding = df_train[column].value_counts() / len(df_train)
            df_train[column] = df_train[column].map(freq_encoding)
            df_test[column] = df_test[column].map(freq_encoding)
            df_test[column] = df_test[column].fillna(0)
            # Save the frequency encoding map
            self.frequency_encoding_maps[column] = freq_encoding

        for col in high_cardinality_features:
            frequency_encoding(col, X_train, X_test)
            categorical_features.remove(col)

        # One-hot encoding for remaining categorical features
        if categorical_features:
            X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

            # Align train and test sets (features only)
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # Update numerical columns after encoding
        self.numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target_column_name in self.numerical_cols:
            self.numerical_cols.remove(self.target_column_name)

        # Combine features and target
        train_df_transformed = X_train.copy()
        train_df_transformed[self.target_column_name] = Y_train.values

        test_df_transformed = X_test.copy()
        test_df_transformed[self.target_column_name] = Y_test.values
        
        # Final check for NaN values in transformed data
        print("\nFinal check for NaN values in transformed training data:")
        print(train_df_transformed.isna().sum())
        print("\nFinal check for NaN values in transformed testing data:")
        print(test_df_transformed.isna().sum())

        partitioned_data = [train_df_transformed, test_df_transformed]
        
        # Save the frequency encoding maps to a file
        joblib.dump(self.frequency_encoding_maps, 'frequency_encoding_maps.joblib')

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