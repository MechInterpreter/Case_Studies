import json
import logging
import os
import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from raw_data_handler import Raw_Data_Handler
from dataset_design import Dataset_Designer
from feature_extractor import Feature_Extractor

class Pipeline():
    def __init__(self, threshold, version: str = 'Dummy'):
        self.threshold = threshold
        self.version = version
        #self.model_path = f'securebank/storage/models/artifacts/{version}.joblib'
        self.model_path = f'storage/models/artifacts/{version}.joblib'
        self.model = self.load_model()
        self.history = [] # For prediction history
        self.training_history = []  # For training performance metrics
        self.feature_names = []
    
    def load_model(self):
        if os.path.exists(self.model_path) and self.version != 'Dummy':
            try:
                model = joblib.load(self.model_path)
                logging.info(f"Loaded model from {self.model_path}")
                return model
            except Exception as e:
                logging.error(f"Failed to load model from {self.model_path}: {e}")
                raise e
        elif self.version == 'Dummy':
            logging.warning(f"Model file not found at {self.model_path}. Initializing a new RandomForestClassifier.")
            model = RandomForestClassifier(
                n_estimators=200,
                criterion='entropy',
                min_samples_split=10,
                bootstrap=True, 
                min_samples_leaf=4,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            logging.info("Initialized a new RandomForestClassifier.")
            return model
        else:
            logging.error(f"Model file not found at {self.model_path} and version is not 'Dummy'.")
            raise FileNotFoundError(f"Model file not found at {self.model_path}.")
        
    # Function to read different file types based on the file extension
    def read_file(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            with open(file_path) as reader:
                json_data = json.load(reader)
            df = pd.json_normalize(json_data)  # Flatten JSON file in case of nested structures
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        self.df = df
        return df
    
    def process(self, csv, parquet, json, sample_fraction=None):
        # Load and transform data
        train_transformed, test_transformed = self.load_data(csv, parquet, json)

        # Sample and balance data
        X_train_resampled, Y_train_resampled, X_test_final, Y_test_final = self.smote_data(
            train_transformed, test_transformed, sample_fraction
        )

        # Train and evaluate model
        self.train_eval_model(X_train_resampled, Y_train_resampled, X_test_final, Y_test_final)    

    def predict(self, input_data: dict) -> bool:
        logging.info(f"Received input data for prediction: {input_data}")
        
        # Convert input_data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Align features with the model
        df_aligned = self.align_features(df)
        
        print(df_aligned)
        
        # Perform prediction
        prediction = self.model.predict(df_aligned)

        # Store prediction in history
        self.history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'input': input_data,
            'prediction': int(prediction[0])
        })
        
        prediction_output = bool(prediction[0])

        return prediction_output  # 0: legitimate, 1: fraud
    
    def create_data(self, csv, parquet, json, sample_fraction=None, apply_smote=True):
        # Load and transform data
        train_transformed, _ = self.load_data(csv, parquet, json)

        # Optionally sample and balance data
        if apply_smote:
            X_train_resampled, Y_train_resampled, _, _ = self.smote_data(
                train_transformed, train_transformed, sample_fraction
            )
        else:
            # If not applying SMOTE, just sample the data
            X_train = train_transformed.drop(columns=['fraudulence'])
            Y_train = train_transformed['fraudulence']

            if sample_fraction:
                X_train_resampled, _, Y_train_resampled, _ = train_test_split(
                    X_train, Y_train,
                    train_size=float(sample_fraction),
                    stratify=Y_train,
                    random_state=42
                )
            else:
                X_train_resampled = X_train
                Y_train_resampled = Y_train

        # Combine features and target
        generated_dataset = X_train_resampled.copy()
        generated_dataset['fraudulence'] = Y_train_resampled

        # Save the generated dataset
        generated_dataset.to_parquet('generated_training_data.parquet')
        print("New training dataset generated and saved to 'generated_training_data.parquet'.")

        return generated_dataset
            
    def load_data(self, csv, parquet, json):
        # Raw Data Handler
        rdh = Raw_Data_Handler()
        e, x, t = rdh.extract(csv, parquet, json)
        transformed = rdh.transform(e, x, t)
        rdh.load(output_filename=r'output_data.parquet')
        
        # Dataset Designer
        dsd = Dataset_Designer(test_size=0.2, target_column_name='fraudulence')
        extracted = dsd.extract('output_data.parquet')
        partitioned = dsd.sample(extracted)
        dsd.load('partitioned.parquet')
        
        # Feature Extractor
        fe = Feature_Extractor(target_column_name='fraudulence')
        train, test = fe.extract('partitioned.parquet')
        fe_trans = fe.transform(train, test)
        
        train_transformed = fe_trans[0]
        test_transformed = fe_trans[1]

        return train_transformed, test_transformed
    
    def smote_data(self, train_transformed, test_transformed, sample_fraction=None):
        # Separate features and target
        X_train = train_transformed.drop(columns=['fraudulence'])
        Y_train = train_transformed['fraudulence']
        X_test = test_transformed.drop(columns=['fraudulence'])
        Y_test = test_transformed['fraudulence']
        
        # Initialize SMOTE
        smote = SMOTE(random_state=42, sampling_strategy='auto', n_jobs=-1)
        
        if sample_fraction:
            X_train_sampled, _, Y_train_sampled, _ = train_test_split(
                X_train, Y_train,
                train_size=float(sample_fraction),
                stratify=Y_train,
                random_state=42
            )

            # Sample the testing data if it's too large
            X_test_sampled, _, Y_test_sampled, _ = train_test_split(
                X_test, Y_test,
                train_size=float(sample_fraction),
                stratify=Y_test,
                random_state=42
            )
            print(f"Original Training Set Size: {X_train.shape[0]}")
            print(f"Sampled Training Set Size: {X_train_sampled.shape[0]}")
            print(f"Original Testing Set Size: {X_test.shape[0]}")
            print(f"Sampled Testing Set Size: {X_test_sampled.shape[0]}")
            print(X_train_sampled)
            print(Y_train_sampled)
            
            # Apply SMOTE to the training data
            X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_sampled, Y_train_sampled)
            
            # Reindex the test set to match the training set's column order
            X_test_final = X_test_sampled.reindex(columns=X_train_resampled.columns)
            Y_test_final = Y_test_sampled
            
        else:
            # Apply SMOTE to the training data
            X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
            
            # Reindex the test set to match the training set's column order
            X_test_final = X_test.reindex(columns=X_train_resampled.columns)
            Y_test_final = Y_test
        
        return X_train_resampled, Y_train_resampled, X_test_final, Y_test_final

    def train_eval_model(self, X_train_resampled, Y_train_resampled, X_test_final, Y_test_final):
        # Configure logging
        logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Verify the class distribution after SMOTE
        print("Y_train Class distribution after SMOTE:")
        print(Y_train_resampled.value_counts())
        
        # Define the directory to save the models
        save_dir = 'securebank/storage/models/artifacts'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Train the model
        print(f"Training {self.version}...")
        self.model.fit(X_train_resampled, Y_train_resampled)

        # Save the trained model
        joblib.dump(self.model, self.model_path)
        print(f"{self.version} saved to {self.model_path}")

        # Predict on the final test set
        y_proba = self.model.predict_proba(X_test_final)[:, 1]
        pred = (y_proba >= self.threshold).astype(int)

        # Generate classification report and confusion matrix
        accuracy = accuracy_score(Y_test_final, pred)
        report = classification_report(Y_test_final, pred)
        matrix = confusion_matrix(Y_test_final, pred)

        # Print the accuracy, classification report, and confusion matrix
        print(f"{self.version} - Accuracy: {accuracy:.4f}")
        print(f"{self.version} - Classification Report:\n{report}")
        print(f"{self.version} - Confusion Matrix:\n{matrix}\n")


        # Performance Auditing Section


        # Compute Precision-Recall Curve and AUC-PR
        precision, recall, thresholds_pr = precision_recall_curve(Y_test_final, y_proba)
        auc_pr = average_precision_score(Y_test_final, y_proba)

        # Plot Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label=f'PR Curve (AUC = {auc_pr:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        # Save the Precision-Recall Curve plot
        plt.savefig(f'{self.version}_precision_recall_curve.png')
        plt.close()
        print(f"Precision-Recall curve saved as '{self.version}_precision_recall_curve.png'.")

        # Print AUC-PR value
        print(f"AUC-PR (Precision-Recall AUC): {auc_pr:.2f}")

        # Find the Optimal Threshold for PR Curve

        # Compute the F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Adjust thresholds_pr to match the length of f1_scores
        thresholds_pr = np.append(thresholds_pr, 1.0)  # Add 1.0 to match lengths

        # Find the index of the maximum F1 score
        optimal_idx_pr = np.argmax(f1_scores)

        # Get the optimal threshold for PR
        optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

        # Print the optimal threshold for PR
        print(f"Optimal Threshold for PR (Maximizing F1 Score): {optimal_threshold_pr:.4f}")

        # Visualize Optimal Threshold on Precision-Recall Curve

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label=f'PR Curve (AUC = {auc_pr:.2f})')
        plt.scatter(
            recall[optimal_idx_pr],
            precision[optimal_idx_pr],
            color='green',
            marker='o',
            label=f'Optimal Threshold = {optimal_threshold_pr:.4f}\n(Precision = {precision[optimal_idx_pr]:.2f}, Recall = {recall[optimal_idx_pr]:.2f})',
            s=100
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Optimal Threshold')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'{self.version}_precision_recall_curve_with_optimal_threshold.png')
        plt.close()
        print(f"Precision-Recall curve with optimal threshold saved as '{self.version}_precision_recall_curve_with_optimal_threshold.png'.")

        # Save the classification report to a text file
        with open(f'{self.version}_classification_report.txt', 'w') as f:
            f.write(report)
        print(f"Classification report saved as '{self.version}_classification_report.txt'.")

        # Save the confusion matrix to a text file
        np.savetxt(f'{self.version}_confusion_matrix.txt', matrix, fmt='%d')
        print(f"Confusion matrix saved as '{self.version}_confusion_matrix.txt'.")

        # Log metrics
        logging.info(f"Model Version: {self.version}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"AUC-PR: {auc_pr:.4f}")
        logging.info(f"Optimal Threshold: {optimal_threshold_pr:.4f}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{matrix}")

        # Update the threshold based on optimal value
        self.threshold = optimal_threshold_pr
        print(f"Threshold updated to optimal value: {self.threshold:.4f}")
        
        # Store training performance metrics and related information
        training_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'version': self.version,
            'accuracy': accuracy,
            'auc_pr': auc_pr,
            'optimal_threshold': self.threshold,
            'classification_report': report,
            'confusion_matrix': matrix.tolist(),  # Convert numpy array to list for serialization
            'precision_recall_curve': f'{self.version}_precision_recall_curve.png',
            'pr_curve_with_threshold': f'{self.version}_precision_recall_curve_with_optimal_threshold.png',
            'classification_report_file': f'{self.version}_classification_report.txt',
            'confusion_matrix_file': f'{self.version}_confusion_matrix.txt',
            'model_path': self.model_path,
        }

        # Append the record to the training history
        self.training_history.append(training_record)

        # Return the model and performance metrics
        return self.model, accuracy, report, matrix

    # Align the input features with the model's training features
    def align_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Hardcode the feature names
        model_features = ['unix_time', 'merchant', 'amt', 'merch_lat', 'merch_long',
                          'transaction_hour', 'time_since_last_trans']
        
        # Copy the input data to avoid modifying the original DataFrame
        input_data = input_data.copy()

        # Convert 'trans_date_trans_time' to datetime and extract 'transaction_hour'
        if 'trans_date_trans_time' in input_data.columns:
            input_data['trans_date_trans_time'] = pd.to_datetime(input_data['trans_date_trans_time'], errors='coerce')
            input_data['transaction_hour'] = input_data['trans_date_trans_time'].dt.hour
            # Drop 'trans_date_trans_time' as it's no longer needed
            input_data = input_data.drop(columns=['trans_date_trans_time'])
        else:
            # If 'transaction_hour' is missing, set it to a default value
            if 'transaction_hour' not in input_data.columns:
                input_data['transaction_hour'] = 0

        # Handle 'time_since_last_trans'
        if 'time_since_last_trans' not in input_data.columns:
            input_data['time_since_last_trans'] = 0  # Default value
            
        # Load the scaler
        try:
            scaler = joblib.load('scaler.joblib')
        except FileNotFoundError:
            print("Scaler not found. Please ensure the scaler is saved during training.")
            # Handle the error appropriately
            scaler = None
            
        # Load frequency encoding maps
        try:
            frequency_encoding_maps = joblib.load('frequency_encoding_maps.joblib')
        except FileNotFoundError:
            print("Frequency encoding maps not found. Please ensure they are saved during training.")
            frequency_encoding_maps = {}
            
        # Handle 'merchant' categorical variable
        if 'merchant' in input_data.columns:
            # Apply the saved frequency encoding mapping
            merchant_encoding_map = frequency_encoding_maps.get('merchant', {})
            input_data['merchant'] = input_data['merchant'].map(merchant_encoding_map)
            input_data['merchant'] = input_data['merchant'].fillna(0)
        else:
            # If 'merchant' is missing, add it with a default value
            input_data['merchant'] = 0

        # Drop unnecessary columns
        columns_to_drop = ['cc_num', 'category', 'dob', 'trans_num', 'first', 'last',
                           'city', 'state', 'zip', 'lat', 'long', 'job', 'street', 'city_pop']
        input_data = input_data.drop(columns=columns_to_drop, errors='ignore')
        print("Columns after dropping unnecessary columns:", input_data.columns.tolist())

        # Ensure all model features are present
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        # Remove any extra features not used by the model
        input_data = input_data[model_features]
        print("Columns after removing extra features:", input_data.columns.tolist())

        # Ensure all data is numeric
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        print("Columns after numerical conversion:", input_data.columns.tolist())

        
        # Apply the scaler to numerical features
        numerical_cols = ['unix_time', 'amt', 'merch_lat', 'merch_long', 'time_since_last_trans']
        if scaler is not None:
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        print("Columns after scaling numerical features:", input_data.columns.tolist())

        return input_data

    def get_history(self, prediction_history_file: str = 'prediction_history.csv', training_history_file: str = 'training_history.csv'):
        # Save prediction history
        if self.history:
            prediction_history_df = pd.DataFrame(self.history)
            prediction_history_df.to_csv(prediction_history_file, index=False)
            print(f'Prediction history saved to {prediction_history_file}')
        else:
            print("No prediction history available.")

        # Save training history
        if self.training_history:
            training_history_df = pd.DataFrame(self.training_history)
            training_history_df.to_csv(training_history_file, index=False)
            print(f'Training history saved to {training_history_file}')
        else:
            print("No training history available.")

        # Return combined history
        combined_history = {
            'prediction_history': self.history,
            'training_history': self.training_history
        }
        return combined_history