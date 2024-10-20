import cv2 as cv
import numpy as np

def filter(bboxes, class_ids, scores, score_threshold, nms_iou_threshold):
    indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold=score_threshold, nms_threshold=nms_iou_threshold)
    
    # Lists to store bounding box, class ID, score, and labels
    final_bboxes = []
    final_class_ids = []
    final_scores = []
    final_labels = []
    
    class_labels = ['barcode', 'car', 'cardboard box', 'fire', 'forklift', 'freight container', 'gloves', 
                    'helmet', 'ladder', 'license plate', 'person', 'qr code', 'road sign', 'safety vest', 
                    'smoke', 'traffic cone', 'traffic light', 'truck', 'van', 'wood pallet']
    
    if len(indices) > 0:
        for i in indices:
            # Get the class label from class_ids and class_labels
            label = class_labels[class_ids[i]]
            
            # Append bounding box, class ID, score, and labels
            final_bboxes.append(bboxes[i])
            final_class_ids.append(class_ids[i])
            final_scores.append(scores[i])
            final_labels.append(label)
    
    return final_bboxes, final_class_ids, final_labels, final_scores