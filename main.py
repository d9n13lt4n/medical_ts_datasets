import numpy as np
import pandas as pd

import tensorflow_datasets as tfds
import medical_ts_datasets

########################################
# Preprocess PhysioNet 2019 Dataset
########################################

raw_19 = tfds.load(name='physionet2019', split='train')
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
df.to_csv("output/physionet2019_train.csv", index=False)

raw_19 = tfds.load(name='physionet2019', split='test')
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
df.to_csv("output/physionet2019_test.csv", index=False)

########################################
# Preprocess PhysioNet 2012 Dataset
########################################

raw_12 = tfds.load(name='physionet2012', split='train')
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
df.to_csv("output/physionet2012_train.csv", index=False)

raw_12 = tfds.load(name='physionet2012', split='test')
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
df.to_csv("output/physionet2012_test.csv", index=False)

