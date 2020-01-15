# NNSKGE
Non-negative Sparse Knowledge Graph Embeddings

**Idea**

Apply Non-Negative Tensor Factorization to Knowledge Graphs to get interpretable embeddings.

**Why**

We want embeddings to have a few characteristics to be interpretable:

- To be explainable by a few factors -> **SPARSITY**
- To be efficient in the information they provide, i.e. we don't need to know that dogs don't have wheels -> **NON-NEGATIVITY**

**Challenges/Contributions**

- Not done for KGs
- Implies Tensor factorization
- There are no interpretable KGE models
- We need to find a way to improve sparsity


Code from this repository was obtained from [here](https://github.com/ibalazevic/TuckER). I thank the authors for providing their code.