import json
import os
from pathlib import Path
from collections import defaultdict
import argparse


def convert_bdd_to_yolo(json_file_path, output_dir, image_dir=None):
    """
    Convert BDD100K JSON annotations to YOLO format text files

    Args:
        json_file_path (str): Path to the BDD100K JSON annotation file
        output_dir (str): Directory to save YOLO format text files
        image_dir (str, optional): Directory containing images (for validation)
    """
    # BDD100K class mapping to YOLO class indices
    class_dict = {
        "bus": 0,
        "traffic light": 1,
        "traffic sign": 2,
        "person": 3,
        "bike": 4,
        "truck": 5,
        "motor": 6,
        "car": 7,
        "train": 8,
        "rider": 9,
    }

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load JSON annotations
    with open(json_file_path, "r") as f:
        annotations = json.load(f)

    # Count statistics
    stats = defaultdict(int)
    missing_images = []

    # Process each image annotation
    for annotation in annotations:
        image_name = annotation["name"]
        image_width = annotation.get("width", 1280)  # Default BDD100K width
        image_height = annotation.get("height", 720)  # Default BDD100K height

        # Check if image exists (if image_dir provided)
        if image_dir:
            image_path = Path(image_dir) / image_name
            if not image_path.exists():
                missing_images.append(image_name)
                continue

        # Prepare YOLO format content
        yolo_lines = []

        # Process each label in the image
        for label in annotation.get("labels", []):
            if "box2d" not in label:
                continue  # Skip non-bounding box annotations

            category = label["category"]
            if category not in class_dict:
                continue  # Skip unknown categories

            # Get bounding box coordinates
            box = label["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (x1 + x2) / (2 * image_width)
            y_center = (y1 + y2) / (2 * image_height)
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            # Ensure values are within [0, 1] range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # Get class ID
            class_id = class_dict[category]

            # Format line for YOLO
            yolo_line = (
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            yolo_lines.append(yolo_line)

            # Update statistics
            stats[category] += 1

        # Write YOLO format file
        output_file = output_path / f"{Path(image_name).stem}.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(yolo_lines))

    # Print statistics
    print("Conversion completed!")
    print(f"Processed {len(annotations)} images")
    print(f"Generated {len(list(output_path.glob('*.txt')))} label files")
    print("\nObject counts by category:")
    for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")

    if missing_images:
        print(
            f"\nWarning: {len(missing_images)} images were missing in the image directory"
        )
        if len(missing_images) < 10:  # Only show a few examples
            print("Missing images:", missing_images[:5])
            if len(missing_images) > 5:
                print(f"... and {len(missing_images) - 5} more")


def create_dataset_yaml(output_dir, class_dict, train_split=0.8):
    """
    Create a YAML dataset configuration file for YOLO training

    Args:
        output_dir (str): Directory where the dataset is stored
        class_dict (dict): Dictionary mapping class names to IDs
        train_split (float): Proportion of data to use for training
    """
    # Get all image files
    image_files = list(Path(output_dir).parent.glob("images/*.jpg")) + list(
        Path(output_dir).parent.glob("images/*.png")
    )

    # Split into train and validation
    num_train = int(len(image_files) * train_split)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # Write train and validation text files
    with open(Path(output_dir).parent / "train.txt", "w") as f:
        for img_path in train_files:
            f.write(f"{img_path}\n")

    with open(Path(output_dir).parent / "val.txt", "w") as f:
        for img_path in val_files:
            f.write(f"{img_path}\n")

    # Create YAML content
    yaml_content = f"""# BDD100K Dataset YAML
path: {Path(output_dir).parent}  # dataset root directory
train: train.txt  # train images (relative to 'path')
val: val.txt  # val images (relative to 'path')

# Classes
nc: {len(class_dict)}  # number of classes
names: {list(class_dict.keys())}  # class names
"""

    # Write YAML file
    yaml_path = Path(output_dir).parent / "bdd100k.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nCreated dataset configuration file: {yaml_path}")


def main():
    # parser = argparse.ArgumentParser(description='Convert BDD100K JSON annotations to YOLO format')
    ##parser.add_argument('--json', type=str, required=True, help='Path to BDD100K JSON annotation file')
    ##parser.add_argument('--output', type=str, required=True, help='Output directory for YOLO format files')
    # parser.add_argument('--image-dir', type=str, help='Directory containing images (for validation)')
    # parser.add_argument('--create-yaml', action='store_true', help='Create YAML dataset configuration file')

    # args = parser.parse_args()

    # Class mapping
    class_dict = {
        "bus": 0,
        "traffic light": 1,
        "traffic sign": 2,
        "person": 3,
        "bike": 4,
        "truck": 5,
        "motor": 6,
        "car": 7,
        "train": 8,
        "rider": 9,
    }

    # add calls for val data set as well
    json = "/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    image_dir = "/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_images_100k/bdd100k/images/100k/train/"
    output = image_dir

    # Convert annotations
    convert_bdd_to_yolo(json, output, image_dir)

    # Create YAML if requested
    # if args.create_yaml:
    create_dataset_yaml(output, class_dict)


if __name__ == "__main__":
    main()
