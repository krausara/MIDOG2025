import pandas as pd

midog = pd.read_csv('MIDOG25.csv')
midog_filename = midog['filename'].tolist()
midog_image_id = midog['image_id'].tolist()
midog_majority = midog['majority'].tolist()

midog = pd.DataFrame(data={'image_id': midog_image_id, 'filename': midog_filename, 'majority': midog_majority})

octo = pd.read_csv('OMG-Octo.csv')

merged = pd.concat([midog, octo], ignore_index=True)
merged.to_csv('midog+octo.csv', index=False)
