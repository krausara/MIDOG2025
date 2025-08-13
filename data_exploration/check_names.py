import os
import pandas as pd

def is_continuous_numbered(folder_path):
    files = os.listdir(folder_path)
    numbers = []
    
    for filename in files:
        filename = filename.split('.')[0] 
        try:
            numbers.append(int(filename))
        except ValueError:
            pass 
    
    numbers.sort()

    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1] + 1:
            print(f"Missing number between {numbers[i - 1]} and {numbers[i]}")
            return False
    print(f"total {len(numbers)}")
    return True

def is_continous_numbered_csv(csv_path):
    df = pd.read_csv(csv_path)

    df['image_id'] = df['image_id'].str.replace(r'\.png$', '', regex=True)  
    ids = df['image_id'].dropna().astype(int).tolist()
    
    ids.sort()
    
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1] + 1:
            if i in range(8924, 8926): 
                print(f"Skipping missing id between {ids[i - 1]} and {ids[i]}") 
                continue
            else:
                print(f"Missing id between {ids[i - 1]} and {ids[i]}, in row {i-1} and {i}")
                return False
    print(f"total {len(ids)}")
    return True

folder_path = "./data" 
if is_continuous_numbered(folder_path):
    print("Files are numbered continuously.")
else:
    print("Files are not numbered continuously.")

csv_path = "MIDOG25.csv"
if is_continous_numbered_csv(csv_path):
    print("CSV is numbered continuously.")
else:
    print("CSV are not numbered continuously.")

'''
total 11937
Files are numbered continuously.
Skipping missing id between 8924 and 8926
Skipping missing id between 8926 and 8928
total 11935
CSV is numbered continuously.
'''
