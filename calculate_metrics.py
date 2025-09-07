import os
import numpy as np
from collections import defaultdict


# ------------------------------
# Utility functions
# ------------------------------
def iou(box1, box2):
    """Compute IoU between two boxes (YOLO xywh normalized)."""

    def yolo_to_xyxy(box):
        x, y, w, h = box
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    x1, y1, x2, y2 = yolo_to_xyxy(box1)
    X1, Y1, X2, Y2 = yolo_to_xyxy(box2)

    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    union = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------
# Evaluation
# ------------------------------
def evaluate_yolo(gt_dir, pred_dir, iou_thresholds=None, conf_thresh=0.25):
    """
    Compute detection metrics across IoU thresholds, with per-class precision/recall/F1.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    # Load GT file paths
    gt_files = {
        os.path.splitext(f)[0]: os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.endswith(".txt")
    }

    results = {}

    for iou_thresh in iou_thresholds:
        # Per-class TP, FP, FN
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)

        for fname, gt_path in gt_files.items():
            pred_path = os.path.join(pred_dir, fname + ".txt")

            # Read GT
            gt = []
            with open(gt_path, "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    gt.append([int(cls), x, y, w, h])

            # Read predictions
            preds = []
            if os.path.exists(pred_path):
                with open(pred_path, "r") as f:
                    for line in f:
                        parts = list(map(float, line.split()))
                        if len(parts) == 6:
                            cls, x, y, w, h, conf = parts
                            if conf >= conf_thresh:
                                preds.append([int(cls), x, y, w, h, conf])
                        else:
                            cls, x, y, w, h = parts
                            preds.append([int(cls), x, y, w, h, 1.0])

            preds.sort(key=lambda x: x[-1], reverse=True)

            matched = set()

            for pred in preds:
                p_cls, px, py, pw, ph, conf = pred
                best_iou, best_gt = 0, -1
                for i, g in enumerate(gt):
                    if i in matched:
                        continue
                    g_cls, gx, gy, gw, gh = g
                    if p_cls != g_cls:
                        continue
                    iou_val = iou([px, py, pw, ph], [gx, gy, gw, gh])
                    if iou_val > best_iou:
                        best_iou, best_gt = iou_val, i
                if best_iou >= iou_thresh:
                    class_tp[p_cls] += 1
                    matched.add(best_gt)
                else:
                    class_fp[p_cls] += 1

            # FN for unmatched GT
            for i, g in enumerate(gt):
                if i not in matched:
                    class_fn[int(g[0])] += 1

        # Compute metrics per class
        per_class_metrics = {}
        total_tp = total_fp = total_fn = 0

        for cls in set(list(class_tp.keys()) + list(class_fn.keys())):
            tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            per_class_metrics[cls] = {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "TP": tp,
                "FP": fp,
                "FN": fn,
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Overall (micro-averaged)
        overall_precision = total_tp / (total_tp + total_fp + 1e-6)
        overall_recall = total_tp / (total_tp + total_fn + 1e-6)
        overall_f1 = (
            2
            * overall_precision
            * overall_recall
            / (overall_precision + overall_recall + 1e-6)
        )

        results[iou_thresh] = {
            "Overall": {
                "Precision": overall_precision,
                "Recall": overall_recall,
                "F1": overall_f1,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
            },
            "PerClass": per_class_metrics,
        }

    return results


# ------------------------------
# Example
# ------------------------------
if __name__ == "__main__":
    gt_dir = "/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_images_100k/bdd100k/images/100k/val/"
    pred_dir = "/nfs/ECAC_Data/Somya_data/BDD_Data/Inference_epoch70/labels/"
    results = evaluate_yolo(gt_dir, pred_dir, iou_thresholds=[0.5, 0.75, 0.9])
    for iou_t, vals in results.items():
        print(f"\nIoU {iou_t}:")
        print(" Overall:", vals["Overall"])
        for cls, m in vals["PerClass"].items():
            print(f"  Class {cls}: {m}")
