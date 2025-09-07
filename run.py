import subprocess

# Step 1: Data preprocessing
subprocess.run(["python", "jsonToTxtLabelConversion.py"], check=True)

# Step 2: Train the model
subprocess.run(["python", "train.py"], check=True)

# Step 3: generate results
subprocess.run(["python", "inference.py"], check=True)

#Step 4: Evaluate output and generate metrics
subprocess.run(["python", "calculate_metrics.py"], check=True)
