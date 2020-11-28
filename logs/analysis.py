import pandas as pd
import numpy as np

df = pd.read_csv("./analysis_2000.csv")
gt = np.array(list(df['GT Label']))
fm = np.array(list(df['Fixmatch']))
knn = np.array(list(df['KNN Label']))
names = list(df['Name'])
correct_fm = ((fm - gt) == 0).astype(int)
correct_knn = ((knn - gt) == 0).astype(int)
print(sum(correct_fm))
print(sum(correct_knn))
print(len(gt))
both_correct = sum((correct_fm * correct_knn))
print(both_correct)

