import torch
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse


class BDD100KInference:
    def __init__(
        self, model_path, test_dir, output_dir, conf_threshold=0.25, iou_threshold=0.45
    ):
        """
        Initialize the BDD100K inference class

        Args:
            model_path (str): Path to the trained YOLO model
            test_dir (str): Directory containing test images
            output_dir (str): Directory to save inference results
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.test_dir = test_dir
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)

        # Load model
        self.model = YOLO(model_path)
        print(f"Loaded model from {model_path}")

        # BDD100K class names
        self.class_names = [
            "bus",
            "traffic light",
            "traffic sign",
            "person",
            "bike",
            "truck",
            "motor",
            "car",
            "train",
            "rider",
        ]

        # Color palette for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    def run_inference(self, save_images=True, save_labels=True):
        """
        Run inference on all images in the test directory

        Args:
            save_images (bool): Whether to save images with detections
            save_labels (bool): Whether to save labels in YOLO format
        """
        # Get all test images
        test_images = list(Path(self.test_dir).glob("*.jpg")) + list(
            Path(self.test_dir).glob("*.png")
        )
        print(f"Found {len(test_images)} test images")

        # Initialize results dictionary
        results_dict = {
            "model": str(self.model_path),
            "timestamp": datetime.now().isoformat(),
            "images": [],
        }

        # Process each image
        for i, img_path in enumerate(test_images):
            print(f"Processing image {i+1}/{len(test_images)}: {img_path.name}")

            # Run inference
            results = self.model.predict(
                source=str(img_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,  # We'll save manually to control output location
                verbose=False,
            )

            # Process results
            image_result = self.process_results(img_path, results[0])
            results_dict["images"].append(image_result)

            # Save visualization if requested
            if save_images:
                self.save_visualization(img_path, results[0])

            # Save labels if requested
            if save_labels:
                self.save_labels(img_path, results[0])

        print(f"Inference completed. Results saved to {self.output_dir}")

    def process_results(self, img_path, result):
        """
        Process inference results for a single image

        Args:
            img_path (Path): Path to the image
            result (ultralytics.engine.results.Results): Inference results

        Returns:
            dict: Processed results for the image
        """
        # Get image dimensions
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]

        # Extract detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, class_id) in enumerate(
                zip(boxes, confidences, class_ids)
            ):
                x1, y1, x2, y2 = box

                detection = {
                    "id": i,
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id],
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                    },
                    "bbox_normalized": {
                        "x_center": float((x1 + x2) / (2 * width)),
                        "y_center": float((y1 + y2) / (2 * height)),
                        "width": float((x2 - x1) / width),
                        "height": float((y2 - y1) / height),
                    },
                }
                detections.append(detection)

        # Create image result
        image_result = {
            "image_name": img_path.name,
            "image_path": str(img_path),
            "image_size": {"width": width, "height": height},
            "detections": detections,
            "detection_count": len(detections),
        }

        return image_result

    def save_visualization(self, img_path, result):
        """
        Save image with detection visualizations

        Args:
            img_path (Path): Path to the image
            result (ultralytics.engine.results.Results): Inference results
        """
        # Plot results on image
        plotted_img = result.plot()

        # Save image
        output_path = self.output_dir / "images" / f"det_{img_path.name}"
        cv2.imwrite(str(output_path), plotted_img)

    def save_labels(self, img_path, result):
        """
        Save detections in YOLO format

        Args:
            img_path (Path): Path to the image
            result (ultralytics.engine.results.Results): Inference results
        """
        if result.boxes is not None:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]

            # Extract detections in YOLO format
            labels = []
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box

                # Convert to YOLO format (normalized center coordinates)
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                labels.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}"
                )

            # Save to file
            label_path = self.output_dir / "labels" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

    def create_summary(self, results_dict):
        """
        Create summary statistics from results

        Args:
            results_dict (dict): Results dictionary

        Returns:
            dict: Summary statistics
        """
        # Count detections by class
        class_counts = {class_name: 0 for class_name in self.class_names}
        confidences_by_class = {class_name: [] for class_name in self.class_names}

        for image_result in results_dict["images"]:
            for detection in image_result["detections"]:
                class_name = detection["class_name"]
                class_counts[class_name] += 1
                confidences_by_class[class_name].append(detection["confidence"])

        # Calculate average confidence by class
        avg_confidence_by_class = {}
        for class_name, confidences in confidences_by_class.items():
            if confidences:
                avg_confidence_by_class[class_name] = sum(confidences) / len(
                    confidences
                )
            else:
                avg_confidence_by_class[class_name] = 0

        # Create summary
        summary = {
            "total_images": len(results_dict["images"]),
            "total_detections": sum(class_counts.values()),
            "detections_per_image": sum(class_counts.values())
            / len(results_dict["images"]),
            "class_counts": class_counts,
            "avg_confidence_by_class": avg_confidence_by_class,
            "inference_settings": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
            },
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference on BDD100K Test Set")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--test-dir", type=str, required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_results",
        help="Output directory for results",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--no-images", action="store_true", help="Do not save images with detections"
    )
    parser.add_argument(
        "--no-labels", action="store_true", help="Do not save labels in YOLO format"
    )
    args = parser.parse_args()

    # Initialize inference
    inference = BDD100KInference(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Run inference
    inference.run_inference(
        save_images=not args.no_images,
        save_labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()
