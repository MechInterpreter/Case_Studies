import os
from flask import Flask, request, jsonify
from PIL import Image
import logging
from pipeline import Pipeline
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the pipeline
pipeline = Pipeline(image_size=160, pretrained='casia-webface', device='cpu', index_type='Flat')

# Check and create necessary directories
gallery_directory = 'storage/gallery'
index_file_path = 'storage/catalog/embeddings.index'

if not os.path.exists(gallery_directory):
    logger.info("Creating gallery directory as it does not exist...")
    os.makedirs(gallery_directory)

if not os.path.exists('storage/catalog'):
    logger.info("Creating catalog directory as it does not exist...")
    os.makedirs('storage/catalog')

# Load or build the gallery index before starting the server
if os.path.exists(index_file_path):
    logger.info("Loading gallery index through pipeline...")
    pipeline.load_gallery_index()
else:
    logger.info("Building gallery index through pipeline...")
    pipeline.build_gallery_index(gallery_directory)

# Endpoint to identify top-k nearest neighbors
@app.route("/identify", methods=["POST"])
def identify():
    try:
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        try:
            probe_image = Image.open(file)
            probe_image.verify()  # Ensure the uploaded file is a valid image
            probe_image = Image.open(file)  # Re-open the file after verify
        except (IOError, ValueError) as e:
            return "Invalid image file uploaded", 400

        k = int(request.form.get('k', 5))  # Default k = 5
        if k > len(pipeline.faiss_index.metadata):
            k = len(pipeline.faiss_index.metadata)
            logger.warning(f"Requested k is greater than the number of indexed images, using k={k} instead.")

        # Use the pipeline to perform the search
        results = pipeline.search_gallery(probe_image, k)

        # Convert any NumPy arrays in the results to lists for JSON serialization
        for result in results:
            if isinstance(result['embedding'], np.ndarray):
                result['embedding'] = result['embedding'].tolist()

        # Store the search in the history
        pipeline.history.append({
            'probe_filename': file.filename,
            'k': k,
            'results': results
        })

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in identify endpoint: {e}", exc_info=True)
        return str(e), 500

# Endpoint to add a new identity to the gallery
@app.route("/add", methods=["POST"])
def add_to_gallery():
    try:
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        try:
            image = Image.open(file)
            image.verify()  # Verify the uploaded file is a valid image
            image = Image.open(file)  # Re-open the file after verify
        except (IOError, ValueError) as e:
            return "Invalid image file uploaded", 400

        embedding_vector = pipeline._encode(image)
        metadata = {
            'name': request.form.get('name', 'unknown'),
            'filename': file.filename
        }

        pipeline.faiss_index.add_embeddings(np.array([embedding_vector]), metadata=[metadata])
        # Save updated index
        pipeline.faiss_index.save('storage/catalog/embeddings.index', 'storage/catalog/metadata.pkl')

        return "Image added successfully", 200

    except Exception as e:
        logger.error(f"Error in add_to_gallery endpoint: {e}", exc_info=True)
        return str(e), 500

# Endpoint to retrieve search history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        # Convert any NumPy arrays in history to lists for JSON serialization
        history_serializable = []
        for record in pipeline.history:
            serializable_record = record.copy()
            for result in serializable_record['results']:
                if isinstance(result['embedding'], np.ndarray):
                    result['embedding'] = result['embedding'].tolist()
            history_serializable.append(serializable_record)

        return jsonify(history_serializable)

    except Exception as e:
        logger.error(f"Error in get_history endpoint: {e}", exc_info=True)
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")