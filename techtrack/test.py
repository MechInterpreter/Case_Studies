import os
import cv2 as cv
import numpy as np
from object_detection import Model, draw_bboxes, save_image_with_bboxes, save_predictions_yolo

def test_object_detection():
    # Initialize the model
    cfg_path = "model2.cfg"
    weights_path = "model2.weights"
    model = Model(cfg_path, weights_path)

    # Set up the UDP stream
    udp_stream_url = "udp://127.0.0.1:12345"
    cap = cv.VideoCapture(udp_stream_url)

    if not cap.isOpened():
        print("Error: Unable to open UDP stream.")
        return

    output_dir = "/app/output_directory"
    os.makedirs(output_dir, exist_ok=True)
    
    frames_with_predictions = 0

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame, original_frame = model.preprocess(frame)

        # Perform prediction
        bboxes, class_ids, scores = model.predict(preprocessed_frame, original_frame)

        # Post-process the results
        filtered_bboxes, filtered_class_ids, filtered_scores = model.post_process(bboxes, class_ids, scores, original_frame)

        # Draw bounding boxes
        image_with_boxes = draw_bboxes(original_frame, filtered_bboxes, filtered_class_ids, filtered_scores)

        # Save the image with bounding boxes
        image_filename = os.path.join(output_dir, f"test_frame_{frame_number:04d}.jpg")
        save_image_with_bboxes(image_with_boxes, image_filename)

        # Save predictions in YOLO format
        label_filename = os.path.join(output_dir, f"test_frame_{frame_number:04d}.txt")
        save_predictions_yolo(filtered_bboxes, filtered_class_ids, filtered_scores, original_frame.shape, label_filename)

        if len(filtered_bboxes) > 0:
            frames_with_predictions += 1

        if frame_number % 10 == 0:
            print(f"Processed {frame_number + 1} frames...")

        frame_number += 1

    cap.release()
    print("Object detection test completed successfully.")

if __name__ == "__main__":
    test_object_detection()