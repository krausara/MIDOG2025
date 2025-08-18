import pandas as pd
import os
from sklearn.model_selection import train_test_split

path_split = 'pat_split_all'
if not os.path.exists(path_split) : os.mkdir(path_split)
path_csv = 'merged_data.csv'

# read csv 
midog_df = pd.read_csv(path_csv)
# split the dataset into training and test sets based on unique patient filenames
files_train, files_test = train_test_split(midog_df.filename.unique(), train_size=0.8)

# create a boolean index for training set to filter the dataframe
idxs_train = [x in files_train for x in midog_df.filename.to_list()]
# get the image ids and labels for training set
ids_train = midog_df.image_id[idxs_train].to_list()
fileid_train = [str(i) for i in ids_train] 
labels_train = midog_df.majority[idxs_train].to_list()

# create a boolean index for test set to filter the dataframe
idxs_test = [x in files_test for x in midog_df.filename.to_list()]
# get the image ids and labels for test set
ids_test = midog_df.image_id[idxs_test].to_list()
fileid_test = [str(i) for i in ids_test]
labels_test = midog_df.majority[idxs_test].to_list()

print('TRAINING OVERVIEW:\n')
print('==================\n')
print('Number of patients in train: ',len(files_train),'(%.2f percent)' % (100*len(files_train)/len(midog_df.filename.unique())))
print('Number of patients in test: ',len(files_test),'(%.2f percent)' % (100*len(files_test)/len(midog_df.filename.unique())))

print('\nNumber of NMF in dataset: ',midog_df.majority.value_counts().get('NMF', 0),'(%.2f percent)' % (100*midog_df.majority.value_counts().get('NMF', 0)/midog_df.majority.count()))
print('Number of NMF in train: ',labels_train.count('NMF'),'(%.2f percent)' % (100*labels_train.count('NMF')/len(labels_train)))
print('Number of NMF in test: ',labels_test.count('NMF'),'(%.2f percent)' % (100*labels_test.count('NMF')/len(labels_test)))

print('\nNumber of AMF in dataset: ',midog_df.majority.value_counts().get('AMF', 0),'(%.2f percent)' % (100*midog_df.majority.value_counts().get('AMF', 0)/midog_df.majority.count()))
print('Number of AMF in train: ',labels_train.count('AMF'),'(%.2f percent)' % (100*labels_train.count('AMF')/len(labels_train)))
print('Number of AMF in test: ',labels_test.count('AMF'),'(%.2f percent)' % (100*labels_test.count('AMF')/len(labels_test)))

# save the training and test set as csv files
df_prepared_train = pd.DataFrame(data={'image_id': fileid_train, 'majority': labels_train})
path_df_train = os.path.join(path_split, 'dataset.train.csv')
df_prepared_train.to_csv(path_df_train, sep=',', index=False)

df_prepared_test = pd.DataFrame(data={'image_id': fileid_test, 'majority': labels_test})
path_df_test = os.path.join(path_split, 'dataset.test.csv')
df_prepared_test.to_csv(path_df_test, sep=',', index=False)
