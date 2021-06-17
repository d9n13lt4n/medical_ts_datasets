import numpy as np
import pandas as pd

import tensorflow_datasets as tfds
import medical_ts_datasets

########################################
# Preprocess PhysioNet 2012 Dataset
########################################

raw_12 = tfds.load(name='physionet2012', split='train+test')
raw_12 = list(raw_12.as_numpy_iterator())

data_12 = []

for i in range(len(raw_12)):
    stat = raw_12[i]['combined'][0]
    time = raw_12[i]['combined'][1]
    temp = raw_12[i]['combined'][2]
    label = raw_12[i]['target']
    for t, row in zip(time, temp):
        data_12.append([i, t, *stat, *row, label])

feats = [
    'Index', 'Time', 'Age', 'Gender_0', 'Gender_1', 'Height', 'ICUType_1', 
    'ICUType_2', 'ICUType_3', 'ICUType_4',
    'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
    'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
    'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
    'Urine', 'WBC', 'pH', 'Label'
]

data_12 = np.array(data_12)
print(f"#Features: {len(feats)}")
print(f"data.shape: {data_12.shape}")

df = pd.DataFrame(data_12, columns=feats)

# Merge Gender and ICU Type
df = df.drop("Gender_0", inplace=False, axis=1)
df = df.rename({'Gender_1': 'Gender'}, axis='columns')

icu = df[["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]]
icu_type = []
for index, row in icu.iterrows():
    icu_type.append((np.argmax(row.tolist())+1))

df = df.drop("ICUType_1", inplace=False, axis=1)
df = df.drop("ICUType_2", inplace=False, axis=1)
df = df.drop("ICUType_3", inplace=False, axis=1)
df = df.drop("ICUType_4", inplace=False, axis=1)
df.insert(6, 'ICUType', icu_type)

# Change data type
df['Index'] = df['Index'].apply(np.int64)
df['Label'] = df['Label'].apply(np.int64)

df.to_csv("output/physionet2012.csv", index=False)
