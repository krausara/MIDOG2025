import pandas as pd

# read the three csv files
midog = pd.read_csv('/data/MIDOG25.csv')
midog_filename = midog['filename'].tolist()
midog_image_id = midog['image_id'].tolist()
midog_majority = midog['majority'].tolist()

midog = pd.DataFrame(data={'image_id': midog_image_id, 'filename': midog_filename, 'majority': midog_majority})

amibr = pd.read_csv('/data/AmiBr.csv')
octo = pd.read_csv('/data/OMG-Octo.csv')

# check for duplicate image IDs and keep only one of the duplicates
duplicates = midog[midog.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in MIDOG.")
print(duplicates['image_id'].unique())
midog = midog.drop_duplicates(subset=['image_id'], keep='first')

duplicates = amibr[amibr.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in AmiBr.")
print(duplicates['image_id'].unique())
amibr = amibr.drop_duplicates(subset=['image_id'], keep='first')

duplicates = octo[octo.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in Octo.")
print(duplicates['image_id'].unique())
octo = octo.drop_duplicates(subset=['image_id'], keep='first')

# remove any entries in midog that are in amibr
amibr_filename = amibr['filename'].tolist()
midog = midog[~midog['filename'].isin(amibr_filename)]

# merge the three dataframes
merged = pd.concat([midog, amibr, octo], ignore_index=True)

# print dataset sizes
print(f"Merged dataset has {len(merged)} entries.")
print(f"After removing duplicates, MIDOG has {len(midog)} entries.")
print(f"AmiBr has {len(amibr)} entries.")
print(f"Octo has {len(octo)} entries.")

# print label distribution
print("Label distribution in merged dataset:")
print(merged['majority'].value_counts())

# print unique patients 
print("Unique filenames in merged dataset:")
print(merged['filename'].nunique())

# save merged dataframe to csv
merged.to_csv('merged_data.csv', index=False)