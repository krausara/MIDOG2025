import pandas as pd

midog = pd.read_csv('./MIDOG25.csv')
midog_filename = midog['filename'].tolist()
midog_image_id = midog['image_id'].tolist()
midog_majority = midog['majority'].tolist()

midog = pd.DataFrame(data={'image_id': midog_image_id, 'filename': midog_filename, 'majority': midog_majority})


ami = pd.read_csv('./AmiBr.csv')
amibr_filename = ami['filename'].tolist()

ami = ami[~ami['filename'].isin(midog_filename)]

ami.to_csv('Ami_noMIDOG.csv', index=False)


