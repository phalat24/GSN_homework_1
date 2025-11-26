# Multitask Learning for Geometric Shape Classification and Counting

Link to colab: https://colab.research.google.com/github/phalat24/GSN_homework_1/blob/main/solution/solution.ipynb

This project trains a shared CNN backbone with two heads:
- **135-way classification** of shape configurations (pair of shapes + count split),
- **6D regression** of per-shape counts.

---

## Dataset

From the project root:

```bash
wget https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip
unzip data_gsn.zip -d data
rm data_gsn.zip
```

## Running the experiments

```python
from experiments import (
    run_classification_only,
    run_regression_only,
    run_multitask,
)

trainer_cls, hist_cls = run_classification_only()
trainer_reg, hist_reg = run_regression_only()
trainer_multi, hist_multi = run_multitask(lambda_cnt=0.3)

```