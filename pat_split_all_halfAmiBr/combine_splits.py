import pandas as pd
import matplotlib.pyplot as plt

# plot the distribution of labels in the combined dataset   
def plot_label_distribution(df, title, filename):
    fig, ax = plt.subplots()
    labels = df["majority"].value_counts()
    bar_container = ax.bar(labels.keys(), labels.values)
    ax.set(title=title, ylabel='count', xlabel='Labels')
    ax.bar_label(bar_container, fmt='{:,.0f}')
    plt.savefig(filename)


midog_train = pd.read_csv('pat_split_combined/dataset.train.csv')
midog_test = pd.read_csv('pat_split_combined/dataset.test.csv')
plot_label_distribution(midog_train, "Label Frequency in Combined Train Set", 'pat_split_all_halfAmiBr/midog+octo_train_labels.png')
plot_label_distribution(midog_test, "Label Frequency in Combined Test Set", 'pat_split_all_halfAmiBr/midog+octo_test_labels.png')

omg_train = pd.read_csv('pat_split_amibr/dataset.train.csv')
omg_test = pd.read_csv('pat_split_amibr/dataset.test.csv')
plot_label_distribution(omg_train, "Label Frequency in AmiBr Train Set", 'pat_split_all_halfAmiBr/amibr_train_labels.png')
plot_label_distribution(omg_test, "Label Frequency in AmiBr Test Set", 'pat_split_all_halfAmiBr/amibr_test_labels.png')

combined_train = pd.concat([midog_train, omg_train], ignore_index=True)
combined_test = pd.concat([midog_test, omg_test], ignore_index=True)
plot_label_distribution(combined_train, "Label Frequency in Combined Train Set", 'pat_split_all_halfAmiBr/all_train_labels.png')
plot_label_distribution(combined_test, "Label Frequency in Combined Test Set", 'pat_split_all_halfAmiBr/all_test_labels.png')

combined_train.to_csv('pat_split_all_halfAmiBr/dataset.train.csv', index=False)
combined_test.to_csv('pat_split_all_halfAmiBr/dataset.test.csv', index=False)
