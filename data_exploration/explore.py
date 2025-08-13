import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('MIDOG25.csv')

fig, ax = plt.subplots()
labels = df["majority"].value_counts()
bar_container = ax.bar(labels.keys(), labels.values)
ax.set(title="Label frequence", ylabel='count', xlabel='Labels')
ax.bar_label(bar_container, fmt='{:,.0f}')
plt.savefig('lables.png')

fig, ax = plt.subplots()
tumors = df["Tumor"].value_counts()
bar_container = ax.barh(tumors.keys(), tumors.values)
ax.set(title="tumor frequence", xlabel='count', ylabel='tumors')
ax.bar_label(bar_container, fmt='{:,.0f}')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('tumor_types.png')

fig, ax = plt.subplots()
species = df["Species"].value_counts()
bar_container = ax.bar(species.keys(), species.values)
ax.set(title="Species frequence", ylabel='count', xlabel='Species')
ax.bar_label(bar_container, fmt='{:,.0f}')
plt.savefig('species_types.png')

with open("filenames.txt", "a") as f:
  f.write("unique files: " + str(len(df["filename"].unique())) + "\n")

df["filename"].value_counts().to_csv('filenames.txt', sep=' ', index=True, mode='a')