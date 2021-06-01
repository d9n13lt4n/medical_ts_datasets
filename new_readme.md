# Medical Time Series Datasets

This is a repository for preprocessing medical time-series dataset. This repository was originally forked from this [source](https://github.com/ExpectationMax/medical_ts_datasets). This repository exports the datasets into a `.csv` file for easier access on different frameworks.

## Getting Started

The environment requirements are as follows:
- python: >=3.6.1, <4
- pip: >=19.0.0

Module requirements are as follows:
```
setuptools = ">=41.0.0"
tensorflow-datasets = ">=2.0.0"
tensorflow = ">=1.15.2"
pandas = "*"
```

This repository could be initialized via [poetry](https://python-poetry.org/).
```
# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# Initialize repository
poetry init
```

The directory hierarchy is as follows:
```
|- medical_ts_datasets/
|- scripts/
|- tests/
|- outputs/
|- main.py
```

## Datasets
Currently implemented datasets are:
- ``physionet2012`` (mortality prediction/open access): [link](https://www.physionet.org/content/challenge-2012/1.0.0/)
- ``physionet2019`` (online sepsis early prediction/open access): [link](https://www.physionet.org/content/challenge-2019/1.0.0/)
- ``mimic3_mortality`` (mortality prediction): [demo](https://www.physionet.org/content/mimiciii-demo/1.4/)
- ``mimic3_phenotyping`` (mortality prediction): [demo](https://www.physionet.org/content/mimiciii-demo/1.4/)

The Physionet Competition datasets could be directly obtained using TFDS. MIMIC-III datasets should be downloaded from [physionet.org](https://www.physionet.org/content/mimiciii/1.4/) and preprocessed using this [repo](https://github.com/d9n13lt4n/mimic3-benchmarks) if intended for use.

## Example Usage
This dataset could also be loaded via the `tensorflow_datasets` module. Otherwise, run the `prep_*.py` scripts to obtain the raw `.csv` files.

```python
import tensorflow_datasets as tfds
import medical_ts_datasets

physionet_dataset = tfds.load(name='physionet2012', split='train')
```

For more details, visit the [source](https://github.com/ExpectationMax/medical_ts_datasets) repo.
