import sys, os
import pandas as pd
directory="data/dogcat/train"
results = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        if "dog" in filename:
            results.append([filename, 1])
        else:
            results.append([filename, 0])

df = pd.DataFrame(results)
df.to_csv("data/dogcat.csv")