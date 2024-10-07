import time
import cv2 as cv
import numpy as np
from preprocessing import capture_video

class Model():
    # Load YOLO model with config and weights
    def __init__(self, cfg, weights):
        self.net = cv.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getUnconnectedOutLayersNames()
        
        # Print the output layer names to verify them
        print("Output layer names:", self.ln)
        
        # Define class labels
        self.class_labels = ['barcode', 'car', 'cardboard box', 'fire', 'forklift', 'freight container', 'gloves', 
                             'helmet', 'ladder', 'license plate', 'person', 'qr code', 'road sign', 'safety vest', 
                             'smoke', 'traffic cone', 'traffic light', 'truck', 'van', 'wood pallet']
        
        # Assign unique color to each class for consistency
        np.random.seed(42)  # For reproducibility
        self.colors = np.random.randint(0, 255, size=(len(self.class_labels), 3), dtype='uint8')
    
    # Generator function that yields blobs for each frame
    def frameify(self, filename, drop_rate):
        frames = capture_video(filename, drop_rate)
        if not frames:
            raise FileNotFoundError(f'Video file {filename} could not be opened.')

        for frame in frames:
             yield frame  # Yield original frames
    
    def preprocess(self, frame, input_size=(416, 416)):
        # Save the original frame for later use
        original_frame = frame.copy()
        
        # Resize frame to model's expected input size
        resized_frame = cv.resize(frame, input_size)
        
        # Create blob object from resized_frame
        preprocessed_frame = cv.dnn.blobFromImage(resized_frame, 1/255.0, input_size, swapRB=True, crop=False)
        
        return preprocessed_frame, original_frame
            
    def predict(self, preprocessed_frame):
        # Set preprocessed frame as input to the network
        self.net.setInput(preprocessed_frame)

        # Start timer
        t0 = time.time()

        # Perform forward pass for detections
        outputs = self.net.forward(self.ln)

        # End timer
        t = time.time()

        print(f'Inference time elapsed: {t - t0:.2f} seconds')

        # Extract all detections without applying threshold
        bboxes = []
        scores = []
        class_ids = []
        h, w = preprocessed_frame.shape[2:]  # Get frame dimensions

        for output in outputs:
            for detection in output:
                score = detection[5:]  # Class scores
                class_id = np.argmax(score)  # Predicted class ID
                confidence = score[class_id]  # Confidence score of the predicted class

                # Scale bounding box back to the original image size
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype('int')

                # Calculate top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Append box, score, and class ID to their respective lists
                bboxes.append([x, y, int(width), int(height)])
                scores.append(float(confidence))
                class_ids.append(class_id)

        return bboxes, class_ids, scores

    def post_process(self, boxes, ids, confidences, frame, score_threshold=0.5):
        # Lists to store the filtered results
        filtered_bboxes = []
        filtered_scores = []
        filtered_class_ids = []

        # Get dimensions of the original image
        h, w = frame.shape[:2]

        # Iterate over the bounding boxes, class IDs, and confidence scores
        for box, class_id, confidence in zip(boxes, ids, confidences):
            # Filter out weak predictions by confidence score threshold
            if confidence > score_threshold:
                # Extract box dimensions
                x, y, width, height = box
                # Scale bounding box back to the original image size
                x_min = int(x)
                y_min = int(y)
                x_max = int(x + width)
                y_max = int(y + height)

                # Ensure the bounding box coordinates are within image bounds
                x_min = max(0, min(x_min, w))
                y_min = max(0, min(y_min, h))
                x_max = max(0, min(x_max, w))
                y_max = max(0, min(y_max, h))

                # Append valid bounding boxes, scores, and class IDs to their respective lists
                filtered_bboxes.append([x_min, y_min, x_max, y_max])
                filtered_scores.append(float(confidence))
                filtered_class_ids.append(class_id)

        return filtered_bboxes, filtered_class_ids, filtered_scores

# Function to draw bounding boxes on an image
def draw_bboxes(image, bboxes, class_ids, scores=None):
    class_labels = ['barcode', 'car', 'cardboard box', 'fire', 'forklift', 'freight container', 'gloves', 
                             'helmet', 'ladder', 'license plate', 'person', 'qr code', 'road sign', 'safety vest', 
                             'smoke', 'traffic cone', 'traffic light', 'truck', 'van', 'wood pallet']
    
    colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    
    # For ground truths, create a list of 1.00 (100% confidence) for each bounding box
    if scores is None:
        scores = [1.00] * len(bboxes)
        
    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        x, y, w, h = bbox
        
        # Ensure all coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        label = f"{class_labels[class_id]}: {score:.2f}"
        color = colors[class_id].tolist()
        
        # Draw bounding box
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), color, -1)
        
        # Put label text
        cv.putText(image, label, (x, y - baseline), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image