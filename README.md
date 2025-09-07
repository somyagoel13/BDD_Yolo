1.Download the dataset

2.Convert labels from JSON to YOLO format (labels.txt)

```python jsonToTxtLabelConversion.py```


-> Edit line 190 in jsonToTxtLabelConversion.py to set paths for the JSON file and output label directory.

3.Train the YOLO model

```python train.py```


-> Modify line 22 in train.py to update your dataset path:

data_yaml = (
    "/path/to/your/data.yaml"
)


4.Run inference with the trained model

```python inference.py```


-> Update input/output paths in the script before running.

5.Calculate metrics

```python calculate_metrics.py```


📊 Data Analysis

For dataset exploration and analysis:

jupyter notebook DataAnalysis.ipynb


-> Update the path to JSON files inside the notebook before running.

📦 Requirements

Install all dependencies with:

```pip install -r requirements.txt```

📂 Project Structure
├── DataAnalysis.ipynb          # Notebook for dataset analysis
├── jsonToTxtLabelConversion.py # JSON → YOLO label conversion
├── train.py                    # Model training script
├── inference.py                # Run inference on test data
├── calculate_metrics.py        # Compute accuracy & metrics
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

📸 Sample Results
![det_ca057c8b-e42f0882](https://github.com/user-attachments/assets/3e22e894-23d4-42a8-9fc0-a308a67e66dd)

![det_ca2d1df6-76d59c0c](https://github.com/user-attachments/assets/b6c53b6f-e913-4955-9eb1-1ee54b7eed2a)
