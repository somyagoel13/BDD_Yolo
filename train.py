import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from collections import Counter
from ultralytics import YOLO


def train_yolov8_small_gpu():
    """
    Train YOLOv8 with a small GPU setup.
    Uses smaller batch size, adjusted epochs, and learning rate scheduling.
    """

    # Dataset and model configuration
    data_yaml = (
        "/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_images_100k/"
        "bdd100k/images/100k/bdd100k_final.yaml"
    )
    model_name = "yolov8s.pt"
    epochs = 150  # Increase epochs to compensate for smaller batch size
    imgsz = 640
    batch_size = 4  # Small batch size that fits into GPU memory

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Class distribution (example: BDD dataset)
    class_counts = Counter(
        {
            7: 713211,
            2: 239686,
            1: 186117,
            3: 91349,
            5: 29971,
            0: 11672,
            4: 7210,
            9: 4517,
            6: 3002,
            8: 136,
        }
    )

    # Precomputed class weights for imbalance handling
    class_weights = torch.tensor(
        [
            1.0465e-02,
            6.5627e-04,
            5.0960e-04,
            1.3371e-03,
            1.6941e-02,
            4.0754e-03,
            4.0687e-02,
            1.7126e-04,
            8.9812e-01,
            2.7041e-02,
        ]
    )

    # Print dataset info
    print("Class counts:", class_counts)
    print("Class weights:", class_weights)
    print(f"Using batch size {batch_size}")
    print(f"Training for {epochs} epochs to compensate for small batch size")

    # Load YOLOv8 model
    model = YOLO(model_name)

    # Train the model with adjusted parameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        # Learning rate schedule
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor (lr0 * lrf)
        # Augmentations to help balance classes
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Training behavior
        cos_lr=True,
        patience=50,
        save_period=10,
        # Experiment logging
        project="bdd100k_training",
        name=f"yolov8s_small_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Analyze results after training
    analyze_training_results(results, class_counts)

    return results


def analyze_training_results(results, class_counts):
    """
    Analyze YOLOv8 training results and generate plots + CSV.

    Args:
        results: YOLOv8 training results object.
        class_counts (Counter): Distribution of class instances.
    """
    # Create analysis output directory
    output_dir = Path("training_analysis")
    output_dir.mkdir(exist_ok=True)

    # Locate results CSV
    results_csv = Path(results.save_dir) / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)

        # Set up plotting grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Training Results", fontsize=16)

        # Box loss curves
        axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
        axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Class loss curves
        axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
        axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
        axes[0, 1].set_title("Class Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # mAP metrics
        axes[0, 2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
        axes[0, 2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
        axes[0, 2].set_title("mAP Metrics")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("mAP")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Precision and Recall
        axes[1, 0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
        axes[1, 0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
        axes[1, 0].set_title("Precision & Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate curve
        axes[1, 1].plot(df["epoch"], df["lr/pg0"], label="Learning Rate")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True)

        # Class distribution histogram
        axes[1, 2].bar(range(len(class_counts)), list(class_counts.values()))
        axes[1, 2].set_title("Class Distribution")
        axes[1, 2].set_xlabel("Class ID")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].grid(True)

        # Save training plots
        plt.tight_layout()
        plt.savefig(output_dir / "training_results.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save metrics CSV
        df.to_csv(output_dir / "training_metrics.csv", index=False)

        print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    # Run YOLOv8 training with small GPU configuration
    results = train_yolov8_small_gpu()

    print(f"Training completed. Best model saved at: {results.best}")
