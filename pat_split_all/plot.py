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


train = pd.read_csv('./dataset.train.csv')
test = pd.read_csv('./dataset.test.csv')
plot_label_distribution(train, "Label Frequency in Combined Train Set", 'all_train_labels.png')
plot_label_distribution(test, "Label Frequency in Combined Test Set", 'all_test_labels.png')