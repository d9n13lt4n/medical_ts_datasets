import numpy as np
import pandas as pd

import tensorflow_datasets as tfds
import medical_ts_datasets

########################################
# Preprocess PhysioNet 2019 Dataset
########################################

raw_19 = tfds.load(name='physionet2019', split='train+test')
raw_19 = list(raw_19.as_numpy_iterator())

data_19 = []

for i in range(len(raw_19)):
    stat = raw_19[i]['combined'][0]
    time = raw_19[i]['combined'][1]
    temp = raw_19[i]['combined'][2]
    label = raw_19[i]['target']

    for t, row, y in zip(time, temp, label):
        data_19.append([i, t, *stat, *row, y[0]])

feats = [
    'Index', 'Time', 'Age', 'Gender_0', 'Gender_1', 'HospAdmTime',
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
    'WBC', 'Fibrinogen', 'Platelets', 'Label'
]

data_19 = np.array(data_19)
print(f"#Features: {len(feats)}")
print(f"data.shape: {data_19.shape}")

df = pd.DataFrame(data_19, columns=feats)

# Merge Gender columns
df = df.drop("Gender_0", inplace=False, axis=1)
df = df.rename({'Gender_1': 'Gender'}, axis='columns')

# Change data type
df['Index'] = df['Index'].apply(np.int64)
df['Label'] = df['Label'].apply(np.int64)

df.to_csv("output/physionet2019.csv", index=False)
