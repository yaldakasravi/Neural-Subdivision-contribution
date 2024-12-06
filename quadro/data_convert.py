import os
import subprocess

# Input and output base directories

input_base = "./data_meshes/cartoon_elephant_200/"
output_base = "./data_meshes/quadri_elephant_200/"

# Mapping of subfolder names to their respective numbers
subfolders = {"subd0": 500, "subd1": 1000, "subd2": 2000}

# Iterate over each subfolder
for subfolder, number in subfolders.items():
    input_dir = os.path.join(input_base, subfolder)
    output_dir = os.path.join(output_base, subfolder)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each .obj file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".obj"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            
            # Construct and execute the remesher command
            command = f"./quadriflow -i {input_file} -o {output_file} {number}"
            print(f"Executing: {command}")
            subprocess.run(command, shell=True, check=True)

print("Processing complete!")

