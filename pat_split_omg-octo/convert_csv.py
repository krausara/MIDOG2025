import pandas as pd

df = pd.read_csv('OMG-Octo_orig.csv')

image_file = df['image_file'].tolist()
labels = df['label'].tolist()
patients = []
for img in image_file:
    patients.append(img.split('_', 1)[0])
labels = ['NMF' if label == 'MF' else label for label in labels]

df_prepared = pd.DataFrame(data={'image_id': image_file, 'filename': patients, 'majority': labels})
# remove all labels that are not NMF or AMF
df_prepared = df_prepared[df_prepared['majority'].isin(['NMF', 'AMF'])]
df_prepared.to_csv('OMG-Octo_new.csv', sep=',', index=False)