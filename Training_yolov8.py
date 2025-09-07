import torch
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import yaml

# Custom training function for small GPU
def train_yolov8_small_gpu():
    # Configuration
    data_yaml = '/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_images_100k/bdd100k/images/100k/bdd100k_final.yaml'
    model_name = 'yolov8s.pt'
    epochs = 150  # More epochs to compensate for small batch size
    imgsz = 640
    batch_size = 4  # Small batch size that fits your GPU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Print class information
    class_counts = Counter({7: 713211, 2: 239686, 1: 186117, 3: 91349, 5: 29971, 0: 11672, 4: 7210, 9: 4517, 6: 3002, 8: 136})
    class_weights = torch.tensor([1.0465e-02, 6.5627e-04, 5.0960e-04, 1.3371e-03, 1.6941e-02, 4.0754e-03, 4.0687e-02, 1.7126e-04, 8.9812e-01, 2.7041e-02])
    
    print("Class counts:", class_counts)
    print("Class weights:", class_weights)
    print(f"Using batch size {batch_size}")
    print(f"Training for {epochs} epochs to compensate for small batch size")
    
    # Load model
    model = YOLO(model_name)
    
    # Use the standard training API with adjusted parameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        # Learning rate settings (adjusted for small batch size)
        lr0=0.001,  # Lower initial learning rate
        lrf=0.01,   # Final learning rate (lr0 * lrf)
        # Augmentation to help with class imbalance
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
        # Additional parameters
        cos_lr=True,
        patience=50,
        save_period=10,
        # Project and name for organization
        project='bdd100k_training',
        name=f'yolov8s_small_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    
    # After training, analyze results
    analyze_training_results(results, class_counts)
    
    return results

def analyze_training_results(results, class_counts):
    """Analyze and plot training results"""
    # Create output directory
    output_dir = Path("training_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Load results CSV
    results_csv = Path(results.save_dir) / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Results', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Class loss
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
        axes[0, 1].set_title('Class Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mAP curves
        axes[0, 2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[0, 2].set_title('mAP Metrics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('mAP')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Precision-Recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        
        # Class distribution
        axes[1, 2].bar(range(len(class_counts)), list(class_counts.values()))
        axes[1, 2].set_title('Class Distribution')
        axes[1, 2].set_xlabel('Class ID')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        df.to_csv(output_dir / "training_metrics.csv", index=False)
        
        print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    # Train the model with small batch size
    results = train_yolov8_small_gpu()
    
    print(f"Training completed. Best model saved at: {results.best}")
