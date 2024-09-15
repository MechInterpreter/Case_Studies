from flask import Flask, request, jsonify
from pipeline import Pipeline

app = Flask(__name__)

# Load pipeline with default model version
pipeline = Pipeline(version='Random Forest')

# Define predict endpoint
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
            return jsonify({'error': 'Missing one or more required fields'}), 400

        # Make prediction using Pipeline predict method
        prediction = pipeline.predict(transaction_data)
        
        # Return prediction result
        return jsonify({'prediction': 'fraud' if prediction else 'legitimate'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(port=5000, debug=True)