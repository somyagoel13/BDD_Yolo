Steps for model training :
1. Download the dataset.
2. Convert labels from json to labels.txt (yolo format)
3. train the yolo model
4. generate inference
5. calculate metrics

All the required libraries can be installed with requirement.txt
pip install -r requirements.txt

For Data Analysis :
use the DataAnalysis.ipynb 
You need to change the path to json files. 

For label conversion , change the paths of json file and output label directory at line 190
python jsonToTxtLabelConversion.py

For training model, modify line 22 of train.py
data_yaml = (
        "/nfs/ECAC_Data/Somya_data/BDD_Data/bdd100k_images_100k/"
        "bdd100k/images/100k/bdd100k_final.yaml"
    ) // replace with your paths
run : 
python train.py

For generating results change the paths and run
python inference.py

For computing accuracy :
python calculate_metrics.py
