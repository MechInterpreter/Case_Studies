from flask import Flask, request, jsonify
from pipeline import Pipeline
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Pipeline once at startup
try:
    pipeline = Pipeline(threshold=0.62, version='Trained')
    logging.info("Pipeline initialized and model loaded.")
except Exception as e:
    logging.error(f"Failed to initialize the model: {e}")
    exit(1)  # Exit if the model fails to load

# Define /health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'}), 200

# Define /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data from request
        transaction_data = request.json

        # Ensure required keys are present in request
        required_keys = [
            'trans_date_trans_time', 'cc_num', 'unix_time', 'merchant',
            'category', 'amt', 'merch_lat', 'merch_long'
        ]

        if not all(key in transaction_data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in transaction_data]
            error_message = f'Missing required fields: {", ".join(missing_keys)}'
            logging.warning(error_message)
            return jsonify({'error': error_message}), 400

        # Make prediction using Pipeline's predict method
        prediction = pipeline.predict(transaction_data)
        logging.info(f"Prediction made: {'fraud' if prediction else 'legitimate'}")

        # Return prediction result
        return jsonify({'prediction': 'fraud' if prediction else 'legitimate'}), 200

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)