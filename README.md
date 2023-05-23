# The code of 《Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction》

### The structure of this project：
- data/*. Including the data and the dataloader. First, download the corresponding data and put it into the folder.

  - [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge)
  - [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction)
  - [Malware](https://www.kaggle.com/c/microsoft-malware-prediction)
  - [Frappe](https://www.baltrunas.info/context-aware/frappe)
  - [ML-tag](https://grouplens.org/datasets/movielens/)

- models/*. The implement of existing CTR models and our proposed GCN, GDCN-S and GDCN-P.  There are currently over 30 CTR models available here. We will continue to update and add new models.

- utils/*. Some common functions.

- main_base.py. Run this project by this file.
