import cv2 as cv
import numpy as np

# Function to convert normalized bounding box to absolute coordinates
def denormalize(box, img_width=640, img_height=640):
    x_center, y_center, width, height = box
    
    abs_width = width * img_width
    abs_height = height * img_height
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    
    return [int(x1), int(y1), int(abs_width), int(abs_height)]

# Updated IoU calculation
def calculate_iou(pred_box, gt_box):
    # Extract coordinates of the bounding boxes
    xA1, yA1, wA, hA = pred_box
    xB1, yB1, wB, hB = gt_box
    
    # Calculate bottom-right coordinates of the boxes
    xA2 = xA1 + wA
    yA2 = yA1 + hA
    xB2 = xB1 + wB
    yB2 = yB1 + hB
    
    # Compute the coordinates of the intersection rectangle
    xI1 = max(xA1, xB1)
    yI1 = max(yA1, yB1)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)
    
    # Compute the width and height of the intersection rectangle
    inter_width = max(0, xI2 - xI1)
    inter_height = max(0, yI2 - yI1)
    
    # Compute the area of intersection
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    boxA_area = wA * hA
    boxB_area = wB * hB
    
    # Compute the IoU, handling division by zero
    iou = inter_area / float(boxA_area + boxB_area - inter_area) if (boxA_area + boxB_area - inter_area) > 0 else 0.0
    
    return iou

def calculate_pr(pred_bboxes, pred_class_ids, pred_scores, gt_bboxes, gt_class_ids, num_classes, iou_threshold=0.5):
    # Initialize per-class true positives, false positives, and false negatives
    tp = {class_id: 0 for class_id in range(num_classes)}
    fp = {class_id: 0 for class_id in range(num_classes)}
    fn = {class_id: 0 for class_id in range(num_classes)}
    
    # Initialize sets to keep track of matched ground truth boxes per class
    matched_gt = {class_id: set() for class_id in range(num_classes)}
    
    # Count ground truth boxes per class
    gt_counts = {class_id: 0 for class_id in range(num_classes)}
    for class_id in gt_class_ids:
        gt_counts[class_id] += 1
    
    # Initialize false negatives with all ground truth boxes per class
    for class_id in range(num_classes):
        fn[class_id] = gt_counts[class_id]
    
    # Sort predictions by scores (confidence)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_bboxes = [pred_bboxes[i] for i in sorted_indices]
    pred_class_ids = [pred_class_ids[i] for i in sorted_indices]
    
    # Check each prediction
    for pred_bbox, pred_class_id in zip(pred_bboxes, pred_class_ids):
        best_iou = 0
        best_gt_idx = -1
        # Compare to all ground truth boxes of the same class
        for gt_idx, (gt_bbox, gt_class_id) in enumerate(zip(gt_bboxes, gt_class_ids)):
            if gt_class_id == pred_class_id and gt_idx not in matched_gt[pred_class_id]:
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Determine if prediction is TP or FP
        if best_iou >= iou_threshold:
            tp[pred_class_id] += 1
            fn[pred_class_id] -= 1
            matched_gt[pred_class_id].add(best_gt_idx)
        else:
            fp[pred_class_id] += 1
    
    # Calculate overall precision and recall
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    return precision, recall, tp, fp, fn

# Calculate average precision (AP)
def calculate_ap(precision, recall):
    # If precision and recall are single values, return precision directly
    if isinstance(precision, (float, int)) and isinstance(recall, (float, int)):
        return precision

    # If precision and recall are lists or arrays with single elements
    if len(precision) == 1 and len(recall) == 1:
        return precision[0]
    
    # Append boundary values to ensure coverage
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    # Ensure precision is non-increasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under the precision-recall curve (AP)
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

# Calculate mAP
def calculate_map(aps):
    return np.mean(aps)

# Calculate 11-point interpolated precision
def calculate_11pi(precision, recall):
    recall_levels = np.linspace(0, 1, 11)
    interpolated_precision = []

    for recall_level in recall_levels:
        p_at_r = max(precision[recall >= recall_level]) if np.any(recall >= recall_level) else 0
        interpolated_precision.append(p_at_r)
    
    return np.array(interpolated_precision)

# Calculate per-class precision, recall, and F1-score
def calculate_precision_recall_f1(metrics, class_ids_list=list(range(20))):
    per_class_metrics = {}
    for class_id in class_ids_list:
        tp = metrics['tp'][class_id]
        fp = metrics['fp'][class_id]
        fn = metrics['fn'][class_id]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        per_class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': metrics['gt_counts'][class_id]
        }
    return per_class_metrics

# Specificity per class
def calculate_specificity(metrics, class_ids_list=list(range(20))):
    per_class_specificity = {}
    for class_id in class_ids_list:
        tn = sum(metrics['tp'].values()) - metrics['tp'][class_id]
        fp = metrics['fp'][class_id]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_specificity[class_id] = specificity
    return per_class_specificity

# Function to read the ground truth data from the text file
def get_ground_truth(file_path):
    with open(file_path, 'r') as file:
        ground_truth = []
        for line in file:
            data = line.split()
            class_id = int(data[0])
            bbox = list(map(float, data[1:]))
            ground_truth.append((class_id, denormalize(bbox)))
    return ground_truth