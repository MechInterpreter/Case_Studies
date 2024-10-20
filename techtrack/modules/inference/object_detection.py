import os
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
        # Convert frame to NumPy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Save the original frame for later use
        original_frame = frame.copy()
        
        # Create blob object from frame, including resizing
        preprocessed_frame = cv.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        
        return preprocessed_frame, original_frame
            
    def predict(self, preprocessed_frame, original_frame):
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
        orig_h, orig_w = original_frame.shape[:2]  # Get original frame dimensions
        input_h, input_w = preprocessed_frame.shape[2:]  # Get preprocessed input dimensions

        for output in outputs:
            for detection in output:
                score = detection[5:]  # Class scores
                class_id = np.argmax(score)  # Predicted class ID
                confidence = score[class_id]  # Confidence score of the predicted class
                
                # Extract bounding box center, width, and height from detection
                centerX, centerY, width, height = detection[:4]

                # Scale back the bounding box to the original image size
                x_center_scaled = int(centerX * orig_w)
                y_center_scaled = int(centerY * orig_h)
                width_scaled = int(width * orig_w)
                height_scaled = int(height * orig_h)

                # Calculate top-left and bottom-right corners of the bounding box
                x_min = int(x_center_scaled - (width_scaled / 2))
                y_min = int(y_center_scaled - (height_scaled / 2))
                x_max = int(x_center_scaled + (width_scaled / 2))
                y_max = int(y_center_scaled + (height_scaled / 2))

                # Ensure the bounding box is within the bounds of the original image
                x_min = max(0, min(x_min, orig_w))
                y_min = max(0, min(y_min, orig_h))
                x_max = max(0, min(x_max, orig_w))
                y_max = max(0, min(y_max, orig_h))

                # Append box, score, and class ID to their respective lists
                bboxes.append([x_min, y_min, x_max, y_max])
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
                x_min, y_min, x_max, y_max = box
                
                # Scale bounding box back to the original image size
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)

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

def draw_bboxes(image, bboxes, class_ids, scores=None):
    class_labels = ['barcode', 'car', 'cardboard box', 'fire', 'forklift', 'freight container', 'gloves', 
                            'helmet', 'ladder', 'license plate', 'person', 'qr code', 'road sign', 'safety vest', 
                            'smoke', 'traffic cone', 'traffic light', 'truck', 'van', 'wood pallet']
    
    colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    
    # For ground truths, create a list of 1.00 (100% confidence) for each bounding box
    if scores is None:
        scores = [1.00] * len(bboxes)
    
    # Loop through bounding boxes to draw
    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        x_min, y_min, x_max, y_max = bbox

        # Ensure all coordinates are integers
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        label = f"{class_labels[class_id]}: {score:.2f}"
        color = colors[class_id].tolist()

        # Draw bounding box
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)

        # Put label text
        cv.putText(image, label, (x_min, y_min - baseline), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

def save_image_with_bboxes(image, filename):
    cv.imwrite(filename, image)
    print(f"Image saved as {filename}")

def save_predictions_yolo(bboxes, class_ids, scores, image_shape, filename):
    with open(filename, 'w') as f:
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            x_min, y_min, x_max, y_max = bbox
            
            # Calculate width and height of the bounding box
            w = x_max - x_min
            h = y_max - y_min
            
            # Calculate center of the bounding box
            x_center = x_min + w / 2
            y_center = y_min + h / 2
            
            # Normalize the coordinates (YOLO format)
            x_center /= image_shape[1]  # Divide by image width
            y_center /= image_shape[0]  # Divide by image height
            width = w / image_shape[1]  # Divide by image width
            height = h / image_shape[0]  # Divide by image height
            
            # Write to file in YOLO format
            f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
    
    print(f"Predictions saved as {filename}")