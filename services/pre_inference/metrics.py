import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple

def _calculate_iou(poly1: Polygon, poly2: Polygon) -> float:
    """Calculate the Intersection over Union of two polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return intersection / union

def calculate_geometric_variance_and_consensus(masks: List[List[List[float]]]) -> Tuple[float, List[List[float]]]:
    """
    Calculate geometric variance and determine the consensus mask.
    
    geometric_variance: Mean pairwise IoU distance (1 - IoU).
    consensus_mask: Medoid polygon (highest average IoU to other polygons).
    """
    polygons = []
    for m in masks:
        # Shapely requires at least 3 points for a valid polygon.
        if len(m) >= 3:
            polygons.append(Polygon(m))
        else:
            polygons.append(Polygon())

    n = len(polygons)
    if n == 0:
        return 0.0, []
    
    if n == 1:
        return 0.0, masks[0]

    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                iou_matrix[i][j] = 1.0
            else:
                iou = _calculate_iou(polygons[i], polygons[j])
                iou_matrix[i][j] = iou
                iou_matrix[j][i] = iou

    # Geometric variance = Mean of pairwise (1 - IoU) for the upper triangle (excluding diagonal)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(1.0 - iou_matrix[i][j])
    
    geometric_variance = float(np.mean(distances)) if distances else 0.0

    # Medoid polygon = the one with the highest average IoU with all others
    mean_ious = np.mean(iou_matrix, axis=1)
    medoid_idx = int(np.argmax(mean_ious))
    consensus_mask = masks[medoid_idx]

    return geometric_variance, consensus_mask


def calculate_class_logit_entropy(logits_list: List[List[float]]) -> float:
    """
    Calculate Shannon entropy of the softmax-averaged class logits across MCD samples.
    """
    if not logits_list:
        return 0.0
        
    logits_arr = np.array(logits_list) # shape: (N, num_classes)
    
    # Softmax for each sample
    # Prevent overflow by subtracting max
    shifted_logits = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Average probabilities across the MCD samples
    mean_probs = np.mean(probs, axis=0) # shape: (num_classes,)
    
    # Shannon entropy
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon))
    
    return float(entropy)
