import pandas as pd

df = pd.read_csv('Ami-Br.csv')
patients = df['slide']
labels = df['majority_atypical']

# false to NMF, true to AMF
labels = ['NMF' if label == False else 'AMF' for label in labels]
# image_file setzt sich aus spalte dataset und uid zusammen
image_file = df['dataset'] + '_' + df['uid'].astype(str) + '.png'

df_prepared = pd.DataFrame(data={'image_id': image_file, 'filename': patients, 'majority': labels})
df_prepared.to_csv('AmiBr_new.csv', sep=',', index=False)

