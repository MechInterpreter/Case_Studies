from flask import Flask, request, jsonify
from collaborative import CollaborativeFiltering
from content_based import ContentBasedFiltering
from recommender import Recommender
from diversifier import Diversifier, calculate_intra_list_diversity
from continuous_learning import ContinuousLearner
import pandas as pd
import numpy as np
import json
import os

# Instantiate Flask
app = Flask(__name__)

# Load models
collaborative_model = CollaborativeFiltering(ratings_file='u.data', metadata_file='u.item')
collaborative_model.fit()
content_based_model = ContentBasedFiltering(ratings_file='u.data', metadata_file='u.item')
recommender = Recommender(model=collaborative_model)

# Load users from file
user_file = 'users.json'
if os.path.exists(user_file):
    with open(user_file, 'r') as f:
        users = json.load(f)
else:
    users = []

def save_users():
    with open(user_file, 'w') as f:
        json.dump(users, f)

# Endpoint to add new user
@app.route('/add_user', methods=['POST'])
def add_user():
    user_data = request.get_json()
    user_id = user_data.get('user_id')
    if user_id in users:
        return jsonify({"message": "User already exists"}), 400
    users.append(user_id)
    save_users()
    return jsonify({"message": "User added successfully"}), 200

# Endpoint to get recommendations for a user
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    try:
        recommendations = recommender.rank_items(user_id=user_id, top_n=10)
        return jsonify(recommendations.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to get diverse recommendations for a user
diversifier = Diversifier(
    diversity_measures=[
        lambda selected, genres: calculate_intra_list_diversity(selected, genres),
    ],
    top_n=10,
    metadata_file="u.item"
)

@app.route('/recommend_diverse/<int:user_id>', methods=['GET'])
def recommend_diverse(user_id):
    try:
        initial_recommendations = recommender.rank_items(user_id=user_id, top_n=10)
        reranked_recommendations = diversifier.rerank(initial_recommendations, alpha=0.7)
        return jsonify(reranked_recommendations.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to login
@app.route('/login', methods=['POST'])
def login():
    user_data = request.get_json()
    user_id = user_data.get('user_id')
    if user_id not in users:
        return jsonify({"message": "User not found"}), 404
    return jsonify({"message": "Login successful"}), 200

# Endpoint to check drift and retrain if necessary
baseline_rmse = np.random.normal(0.0, 0.5, 100)
learner = ContinuousLearner(baseline_rmse=baseline_rmse)

@app.route('/check_drift', methods=['POST'])
def check_drift():
    production_rmse = np.random.normal(0.1, 0.5, 200)
    try:
        retrain_needed, p_value = learner.detect_drift(production_rmse, return_pvalue=True)
        if retrain_needed:
            collaborative_model.fit()  # Retrain the model
            return jsonify({"message": "Drift detected, model retrained", "p_value": p_value}), 200
        else:
            return jsonify({"message": "No drift detected", "p_value": p_value}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)