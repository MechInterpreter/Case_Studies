from flask import Flask, request, jsonify
from pipeline import Pipeline
import os

app = Flask(__name__)

# Initialize pipeline
pipeline = Pipeline(embedding_model='all-MiniLM-L6-v2', reranker_type='hybrid')

# Set up a global variable for corpus_directory
corpus_directory = None

# Endpoint to define the corpus directory
@app.route('/set_corpus_directory', methods=['POST'])
def set_corpus_directory():
    global corpus_directory
    try:
        # Get directory path from the request body
        data = request.get_json()
        corpus_directory = data.get('corpus_directory')
        
        if not corpus_directory or not os.path.exists(corpus_directory):
            return jsonify({'error': 'Invalid or missing corpus directory'}), 400

        return jsonify({'message': f'Corpus directory set to {corpus_directory}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to add a single document to the corpus
@app.route('/add_document', methods=['POST'])
def add_document():
    global corpus_directory
    try:
        # Check if the file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file to the corpus_directory
        if corpus_directory:
            file_path = os.path.join(corpus_directory, file.filename)
            file.save(file_path)
            return jsonify({'message': f'File {file.filename} uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Corpus directory not set. Please set it first using /set_corpus_directory'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to preprocess and save the entire corpus
@app.route('/process_and_save_corpus', methods=['POST'])
def process_and_save_corpus():
    global corpus_directory
    try:
        if not corpus_directory:
            return jsonify({'error': 'Corpus directory has not been set. Please set it first using /set_corpus_directory'}), 400

        # Process and add documents to the pipeline's index
        pipeline.preprocess_corpus(corpus_directory)

        # Save the index to disk
        pipeline.save_index(faiss_path="faiss.index", metadata_path="metadata.pkl")

        # Verify if the index is populated
        if hasattr(pipeline.index, 'index') and pipeline.index.index is not None:
            return jsonify({'message': f'Documents processed, indexed, and saved successfully. FAISS index contains {pipeline.index.index.ntotal} items.'}), 200
        else:
            return jsonify({'error': 'Failed to create FAISS index. No items found.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to load the index from saved files
@app.route('/load_index', methods=['POST'])
def load_index():
    try:
        # Get FAISS index and metadata file paths from the request
        data = request.get_json()
        faiss_path = data.get('faiss_path', 'faiss.index')
        metadata_path = data.get('metadata_path', 'metadata.pkl')

        # Load the FAISS index
        pipeline.load_index(faiss_path, metadata_path)

        if hasattr(pipeline.index, 'index') and pipeline.index.index is not None:
            return jsonify({'message': f'Index loaded successfully. FAISS index contains {pipeline.index.index.ntotal} items.'}), 200
        else:
            return jsonify({'error': 'Failed to load FAISS index. No items found.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to ask a question (query) to the system
@app.route('/query', methods=['POST'])
def query():
    try:
        # Check if the index is populated, if not, return an error
        if not hasattr(pipeline.index, 'index') or pipeline.index.index is None or pipeline.index.index.ntotal == 0:
            return jsonify({'error': 'No index found. Please add documents to the corpus first or load the index.'}), 400

        # Get query data from the request
        data = request.get_json()
        query_text = data.get('query')
        k = data.get('k', 5)  # Default to 5 if k is not provided
        rerank = data.get('rerank', True)

        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400

        # Use the pipeline to generate an answer
        answer = pipeline.query(query=query_text, k=k, rerank=rerank)
        return jsonify({'query': query_text, 'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000)