# BDD Yolo


## Index
* [Dataset](#dset)
* [Labeling](#label)
* [Training](#train)
* [Inference](#infer)
* [Evaluation](#eval)
* [Data Analysis](#analyse)
* [Requirements](#req)
* [Project Structure](#pstruct)
* [Sample Results](#res)
* [Overview](https://github.com/somyagoel13/BDD_Yolo/blob/master/BDD_ObjectDetection.pdf)


## <a name="dset"></a> Dataset

Download the dataset using [this](http://bdd-data.berkeley.edu/) link.


## <a name="label"></a> Labeling

Convert labels from JSON to YOLO format (labels.txt)<br/>

```python jsonToTxtLabelConversion.py```
> <b>Note: Edit line 190 in jsonToTxtLabelConversion.py to set paths for the JSON file and output label directory.</b>


## <a name="train"></a> Training

Train the YOLO model<br/>

```python train.py```
> <b>Note: Modify line 22 in train.py to update your dataset path</b>
```
data_yaml = (
    "/path/to/your/data.yaml"
)
```


## <a name="infer"></a> Inference

Run inference with the trained model<br/>

```python inference.py```
><b>Note: Update input/output paths in the script before running.</b>


## <a name="eval"></a> Evaluation

Calculate metrics<br/>

```python calculate_metrics.py```


## <a name="analyse"></a> 📊 Data Analysis

For dataset exploration and analysis:

```jupyter notebook DataAnalysis.ipynb```

><b>Note: Update the path to JSON files inside the notebook before running.</b>


## <a name="req"></a> 📦 Requirements

Install all dependencies with:

```pip install -r requirements.txt```

## <a name="pstruct"></a> 📂 Project Structure
├── DataAnalysis.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Notebook for dataset analysis<br/>
├── jsonToTxtLabelConversion.py&nbsp;# JSON → YOLO label conversion<br/>
├── train.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Model training script<br/>
├── inference.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Run inference on test data<br/>
├── calculate_metrics.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Compute accuracy & metrics<br/>
├── requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dependencies<br/>
└── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Project documentation<br/>

## <a name="res"></a> 📸 Sample Results
![det_ca057c8b-e42f0882](https://github.com/user-attachments/assets/3e22e894-23d4-42a8-9fc0-a308a67e66dd)

![det_ca2d1df6-76d59c0c](https://github.com/user-attachments/assets/b6c53b6f-e913-4955-9eb1-1ee54b7eed2a)
