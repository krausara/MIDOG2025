import pandas as pd

midog = pd.read_csv('MIDOG25.csv')
midog_filename = midog['filename'].tolist()
midog_image_id = midog['image_id'].tolist()
midog_majority = midog['majority'].tolist()

midog = pd.DataFrame(data={'image_id': midog_image_id, 'filename': midog_filename, 'majority': midog_majority})


# find all duplicate image IDs
duplicates = midog[midog.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in MIDOG.")
print(duplicates['image_id'].unique())

amibr = pd.read_csv('AmiBr.csv')
amibr_filename = amibr['filename'].tolist()
# find all duplicate image IDs
duplicates = amibr[amibr.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in AmiBr.")
print(duplicates['image_id'].unique())

octo = pd.read_csv('OMG-Octo.csv')
octo_filename = octo['filename'].tolist()
# find all duplicate image IDs
duplicates = octo[octo.duplicated(subset=['image_id'], keep=False)]
print(f"Found {len(duplicates)} duplicate in Octo.")
print(duplicates['image_id'].unique())
# keep only one of the duplicates
octo = octo.drop_duplicates(subset=['image_id'], keep='first')

# remove any entries in midog that are in amibr or octo
midog = midog[~midog['filename'].isin(amibr_filename)]
midog = midog[~midog['filename'].isin(octo_filename)]

# merge the three dataframes
merged = pd.concat([midog, amibr, octo], ignore_index=True)
merged.to_csv('merged_data.csv', index=False)