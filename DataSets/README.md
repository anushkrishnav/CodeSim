# DataSets 

- PooLC : A Clone detection dataset for python https://huggingface.co/datasets/PoolC/5-fold-clone-detection-600k-5fold
```python
import dask.dataframe as dd

splits = {'train': 'data/train-*-of-*.parquet', 'val': 'data/val-*-of-*.parquet'}
df = dd.read_parquet("hf://datasets/PoolC/5-fold-clone-detection-600k-5fold/" + splits["train"])
```
