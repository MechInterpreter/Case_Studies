import numpy as np
import os
import json
import cv2
from typing import List, Tuple
import torch
import torch.nn.functional as F
from helper import calculate_iou

def compute_yolo_loss(predictions: List[dict], annotations: List[dict], num_classes: int) -> float:
    total_loss = 0.0
    lambda_bb = 5.0
    lambda_obj = 1.0
    lambda_no_obj = 0.5
    lambda_cls = 1.0
    iou_threshold = 0.5

    epsilon = 1e-7  # Small value to prevent log(0)

    for pred in predictions:
        max_iou = 0
        best_match = None

        for ann in annotations:
            iou = calculate_iou(pred['bbox'], ann['bbox'])
            if iou > max_iou:
                max_iou = iou
                best_match = ann

        objectness_pred = torch.tensor([pred['objectness']], dtype=torch.float32).detach()
        if max_iou >= iou_threshold and best_match is not None:
            # Bounding box loss
            bbox_pred = torch.tensor(pred['bbox'], dtype=torch.float32).detach()
            bbox_true = torch.tensor(best_match['bbox'], dtype=torch.float32).detach()
            loc_loss = F.mse_loss(bbox_pred, bbox_true)

            # Classification loss
            class_probs = np.array(pred['class_scores'])  # These are probabilities after softmax and multiplication
            class_probs = np.clip(class_probs / (class_probs.sum() + epsilon), epsilon, 1.0)  # Normalize and clip

            # Use cross-entropy loss directly
            class_pred = torch.tensor(class_probs, dtype=torch.float32).unsqueeze(0).detach()
            class_true = torch.tensor([best_match['label']], dtype=torch.long)

            # Debug statements
            print(f"class_pred shape: {class_pred.shape}, dtype: {class_pred.dtype}")
            print(f"class_true shape: {class_true.shape}, dtype: {class_true.dtype}")
            print(f"class_pred values: {class_pred}")
            print(f"class_true value: {class_true.item()}")

            # Check for NaNs or Infs
            if torch.isnan(class_pred).any() or torch.isinf(class_pred).any():
                print("Warning: class_pred contains NaN or Inf values.")
                continue  # Skip this prediction

            # Use cross-entropy loss for better numerical stability
            cls_loss = F.cross_entropy(class_pred, class_true)

            # Objectness loss
            objectness_true = torch.tensor([1.0], dtype=torch.float32).detach()
            obj_loss = F.mse_loss(objectness_pred, objectness_true)

            total_loss += (
                lambda_bb * loc_loss.item()
                + lambda_cls * cls_loss.item()
                + lambda_obj * obj_loss.item()
            )
        else:
            # Penalize objectness score
            objectness_true = torch.tensor([0.0], dtype=torch.float32).detach()
            obj_loss = F.mse_loss(objectness_pred, objectness_true)
            total_loss += lambda_no_obj * obj_loss.item()
            
    if not predictions:
        print("No predictions available for loss calculation.")
        return 0.0

    return total_loss

def sample_hard_negatives(prediction_dir: str, annotation_dir: str, num_samples: int) -> List[Tuple[str, float]]:
    losses = []

    # List all files in the prediction directory
    pred_files = [f for f in os.listdir(prediction_dir) if f.endswith('.json')]

    for pred_file in pred_files:
        # Load predictions and annotations
        pred_path = os.path.join(prediction_dir, pred_file)
        ann_path = os.path.join(annotation_dir, pred_file)

        with open(pred_path, 'r') as pf:
            predictions = json.load(pf)

        with open(ann_path, 'r') as af:
            annotations = json.load(af)

        # Compute loss
        loss = compute_yolo_loss(predictions, annotations, num_classes=20)
        losses.append((pred_file, loss))

    # Sort losses in descending order (hardest negatives first)
    losses.sort(key=lambda x: x[1], reverse=True)

    # Return the top num_samples hard negatives
    hard_negatives = losses[:num_samples]

    return hard_negatives

def compute_difficulty_metric(image: np.ndarray, predictions: List[dict]) -> float:
    total_entropy = 0.0
    num_predictions = len(predictions)

    for pred in predictions:
        class_probs = np.array(pred['class_scores'])
        # Normalize probabilities if they are not already
        class_probs = np.clip(class_probs / np.sum(class_probs), 1e-6, 1.0)
        entropy = -np.sum(class_probs * np.log(class_probs))
        total_entropy += entropy

    if num_predictions > 0:
        average_entropy = total_entropy / num_predictions
    else:
        average_entropy = 0.0

    return average_entropy