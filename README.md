# CodeSim
Personal Benchmark Collections of current open models in Code similarity for Python Langugage &amp; Personal Experiments


<!-- ## reference
AST: https://til.simonwillison.net/python/tree-sitter

DFG : https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/clonedetection/parser -->


## datasets 
Super Code Clone Detection - 88 (SCD-88)
Creators


Description
SCD-88 is the Python-specific subset of the Cross-Language Clone Detection dataset which was originally extracted from AtCoder, a popular Online Judge. We reformulate this classification task as a retrieval one where given a code and a collection of candidates as the input, the task is to return top-k codes with the same semantic. Models can hence, be evaluated by the MAP@R score. MAP@R is defined as the mean of average precision scores, each of which is evaluated for retrieving R most similar samples given a query. For a code (query), R is the number of other codes in the same class, i.e. R=129 in this dataset. The newly sampled dataset amounts to a total of 11,440 examples where the splits are as follows: 7800 / 1040 / 2600 (Train / Valid / Test).
https://zenodo.org/records/5388452


