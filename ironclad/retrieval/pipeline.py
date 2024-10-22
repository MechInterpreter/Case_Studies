import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from PIL import Image, UnidentifiedImageError
from processing import Preprocessing
from embedding import Embedding
from index import FaissIndex
from search import FaissSearch
import faiss
import pickle
import logging
import time

class Pipeline:
    def __init__(self, image_size=160, pretrained='casia-webface', device='cpu', index_type='Flat'):
        self.preprocessing = Preprocessing(image_size=image_size)
        self.embedding = Embedding(pretrained=pretrained, device=device)
        self.faiss_index = FaissIndex(index_type=index_type)
        self.faiss_search = None  # Will be initialized after index is built
        self.history = []  # List to store search history

    def _encode(self, image):
        preprocessed_image = self.preprocessing.process(image)
        embedding = self.embedding.encode(preprocessed_image)
        return embedding

    def _precompute(self, gallery_directory):
        embeddings = []
        metadata = []

        for root, _, files in os.walk(gallery_directory):
            for file in files:
                image_path = os.path.join(root, file)
                print(f"Attempting to process: {image_path}")
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image = Image.open(image_path)
                        embedding = self.__encode(image)
                        embeddings.append(embedding)
                        metadata.append({
                            'name': os.path.basename(os.path.dirname(image_path)),
                            'filename': file,
                            'path': image_path
                        })
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"Error processing {image_path}: {e}")
                        continue  # Skip this file and continue with the next

        if not embeddings:
            print("No valid images found in the gallery directory.")
            return

        embeddings = np.array(embeddings).astype('float32')
        self.faiss_index.add_embeddings(embeddings, metadata=metadata)
        self.faiss_search = FaissSearch(self.faiss_index)

    def _save_embeddings(self):
        os.makedirs('storage/catalog', exist_ok=True)
        self.faiss_index.save('storage/catalog/embeddings.index', 'storage/catalog/metadata.pkl')

    def search_gallery(self, probe, k):
        try:
            # Ensure k is an integer and greater than 0
            k = max(1, int(k))
            
            # Check if probe is a valid image
            if not isinstance(probe, Image.Image):
                raise ValueError("Probe must be a PIL Image object")

            # Encode the probe image
            probe_embedding = self.__encode(probe)

            # Ensure probe_embedding is a numpy array with correct shape
            if not isinstance(probe_embedding, np.ndarray):
                probe_embedding = np.array(probe_embedding)
            probe_embedding = probe_embedding.reshape(1, -1).astype('float32')

            # Check if the embedding has the correct dimensionality
            if probe_embedding.shape[1] != self.faiss_index.index.d:
                raise ValueError(f"Probe embedding dimensionality ({probe_embedding.shape[1]}) "
                                 f"does not match index dimensionality ({self.faiss_index.index.d})")

            # Perform the search
            logger.info(f"Searching for {k} nearest neighbors")
            distances, indices, metadata = self.faiss_search.search(probe_embedding, k)

            results = []
            for i in range(min(k, len(indices[0]))):
                result = metadata[i].copy()  # Copy the metadata
                result['embedding'] = self.faiss_index.index.reconstruct(int(indices[0][i]))
                result['distance'] = float(distances[0][i])  # Convert to Python float for serialization
                results.append(result)
                
            # Store search record in history
            search_record = {
                'probe_filename': probe.filename if hasattr(probe, 'filename') else 'unknown',
                'k': k,
                'results': results
            }
            self.history.append(search_record)

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in search_gallery: {e}", exc_info=True)
            raise

    def build_gallery_index(self, gallery_directory, stop_at_file=None):
        embeddings = []
        metadata = []
        stop_flag = False

        for root, _, files in os.walk(gallery_directory):
            for file in files:
                # Skip hidden files (those starting with '._')
                if file.startswith('._'):
                    continue
                
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    try:
                        image = Image.open(image_path)
                        embedding = self.__encode(image)
                        embeddings.append(embedding)
                        metadata.append({
                            'name': os.path.basename(os.path.dirname(image_path)),
                            'filename': file,
                            'path': image_path
                        })
                        print(f"Processed: {image_path}")
                        
                        if stop_at_file and file == stop_at_file:
                            print(f"Stopped at {stop_at_file}")
                            stop_flag = True
                            break
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            
            if stop_flag:
                break

        if not embeddings:
            print("No valid images found in the gallery directory.")
            return

        embeddings = np.array(embeddings).astype('float32')
        self.faiss_index.add_embeddings(embeddings, metadata=metadata)
        self.__save_embeddings()

    def load_gallery_index(self):
        self.faiss_index.load('storage/catalog/embeddings.index', 'storage/catalog/metadata.pkl')
        self.faiss_search = FaissSearch(self.faiss_index)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(pipeline, gallery_dir, probe_path, k=5, stop_at_file=None):
    try:
        logger.info("Building gallery index...")
        start_time = time.time()
        pipeline.build_gallery_index(gallery_dir, stop_at_file=stop_at_file)
        logger.info(f"Index building took {time.time() - start_time:.2f} seconds")
        
        logger.info("Loading gallery index...")
        start_time = time.time()
        pipeline.load_gallery_index()
        logger.info(f"Index loading took {time.time() - start_time:.2f} seconds")
        
        logger.info(f"Searching gallery with probe image: {probe_path}")
        probe = Image.open(probe_path)
        
        start_time = time.time()
        results = pipeline.search_gallery(probe, k=k)
        logger.info(f"Search took {time.time() - start_time:.2f} seconds")
        
        logger.info("Search results:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Name: {result['name']}, File: {result['filename']}, Distance: {result['distance']:.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    pipeline = Pipeline()
    gallery_dir = 'storage/gallery'
    probe_path = "simclr_resources/probe/Alan_Ball/Alan_Ball_0002.jpg"
    
    # First attempt: process the entire gallery
    results = run_pipeline(pipeline, gallery_dir, probe_path)
    
    # If the first attempt fails, try again with a stop_at_file
    if results is None:
        logger.info("Retrying with limited gallery processing...")
        results = run_pipeline(pipeline, gallery_dir, probe_path, stop_at_file='Alan_Ball_0002.jpg')
    
    if results is None:
        logger.error("Pipeline execution failed in both attempts.")