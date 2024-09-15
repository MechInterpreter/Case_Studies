import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Pipeline():
    def __init__(self, version: str = 'Random Forest'):
        self.model_path = f'securebank/storage/models/artifacts/{version}.joblib'
        self.model = self.select_model()
        self.history = []
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()

    def predict(self, input_data: dict) -> bool:
        # Convert input_data to a DataFrame
        df = pd.DataFrame([input_data])

        # Perform necessary preprocessing
        df = self.preprocess_data(df)

        # Align features with the model
        df = self.align_features(df)

        # Perform prediction
        prediction = self.model.predict(df)

        # Store prediction in history
        self.history.append({'input': input_data, 'prediction': int(prediction[0])})
        
        prediction_output = bool(prediction[0])

        return prediction_output  # 0: legitimate, 1: fraud

    def select_model(self):
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            print(f'Loaded model from {self.model_path}')
            return model
        else:
            raise FileNotFoundError(f'Model file not found at {self.model_path}')
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle categorical features and unseen categories
        categorical_features = ['sex', 'city', 'state', 'zip', 'job', 'merchant', 'category', 'transaction_day_of_week']
        high_cardinality_features = ['job', 'merchant', 'city', 'zip']
        numerical_features = ['lat', 'long', 'amt', 'merch_lat', 'merch_long']

        # Ensure the categorical features exist in the data before processing
        categorical_features = [col for col in categorical_features if col in df.columns]

        # Handle 'sex' with label encoding
        if 'sex' in df.columns:
            le = LabelEncoder()
            df['sex'] = le.fit_transform(df['sex'].fillna('Unknown'))

        # Frequency encoding for high-cardinality features
        def frequency_encoding(column, df):
            freq_encoding = df[column].value_counts() / len(df)
            df[column] = df[column].map(freq_encoding)
            df[column] = df[column].fillna(0)

        for col in high_cardinality_features:
            if col in df.columns:
                frequency_encoding(col, df)

        # Handle remaining categorical features with One-Hot Encoding
        if categorical_features:
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Ensure the numerical features exist in the data before processing
        numerical_features = [col for col in numerical_features if col in df.columns]

        # Impute missing values for numerical features
        if numerical_features:
            df[numerical_features] = self.numerical_imputer.fit_transform(df[numerical_features])

        # Normalize numerical features
        if numerical_features:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        return df

    # Align the input features with the model's training features
    def align_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        model_features = self.model.feature_names_in_
        input_features = input_data.columns.tolist()

        # Add missing features with a default value
        for feature in model_features:
            if feature not in input_features:
                input_data[feature] = 0  # Fill missing features with 0 or an appropriate default value
        
        # Remove any extra features that the model wasn't trained on
        input_data = input_data[model_features]
        
        return input_data

    def get_history(self, file_path: str = 'prediction_history.csv'):
        # Return history as a dictionary (which it already is)
        history = self.history
        
        # Convert history to a DataFrame and save it to a CSV file
        history_df = pd.DataFrame(history)
        history_df.to_csv(file_path, index=False)
        print(f'History saved to {file_path}')
        
        # Return the history dictionary
        return history