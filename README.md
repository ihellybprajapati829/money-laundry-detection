# Detecting Illicit Transactions (Anti-Money Laundering)

Applied machine learning on the Elliptic Bitcoin transaction dataset to flag illicit transactions. Stage 1 is a tree-based baseline (XGBoost). Stage 2 will test whether the transaction network structure improves detection using a Graph Neural Network.

## Why this problem matters
Serious and organised crime cost Australia up to $60.1 billion in 2020-21, according to Australian Institute of Criminology research cited by AUSTRAC [1]. Money laundering lets these groups profit from and reinvest their proceeds. AUSTRAC's 2024 national risk assessment found that honest businesses are regularly exploited by laundering networks [1][2], and its latest risk update recognises virtual assets and AI as growing enablers of financial crime, including the laundering of scam proceeds [3]. From 1 July 2026, Australia's AML/CTF regime expanded to cover higher-risk professional services such as lawyers, accountants and trust and company service providers [4]. 

Illicit transactions are rare, so a useful detection model must find them without overwhelming investigators with false alarms.

## References (IEEE Referencing Style)
[1] AUSTRAC, "Combating money laundering, terrorism financing and proliferation financing." Accessed: Oct. 6, 2026. [Online]   
    Available: https://www.austrac.gov.au/new-austrac/combating-money-laundering-terrorism-financing-and-proliferation-financing

[2] AUSTRAC, "Money laundering in Australia: National risk assessment 2024,"Jul. 2024. Accessed: Oct. 6, 2026. [Online]. 
    Available: https://www.austrac.gov.au/sites/default/files/2024-07/2024%20AUSTRAC%20Money%20Laundering%20NRA.pdf

[3] AUSTRAC, "AUSTRAC releases updated risk snapshot of Australia's financial crime landscape." Accessed: Oct. 6, 2026. [Online]. 
    Available: https://www.austrac.gov.au/news-and-media/media-release/austrac-releases-updated-risk-snapshot-australias-financial-crime-landscape

[4] Department of Home Affairs, "Overview of the AML/CTF Amendment Act." Accessed: Oct. 6, 2026. [Online]. 
    Available: https://www.homeaffairs.gov.au/criminal-justice/Pages/overview-of-the-amlctf-amendment-act.aspx

## Dataset
##### Elliptic Dataset
##### Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Elliptic Bitcoin dataset: 203,769 transactions (nodes) linked by payment flows (edges), across 49 time steps
- 165 anonymised features per transaction. Per the dataset documentation, the first 93 features represent local information about the transaction, and the remaining 72 features are aggregated features, obtained using transaction information one-hop backward/forward from the center node - giving the maximum, minimum, standard deviation and correlation coefficients of the neighbour transactions for the same information data
- Only 46,564 transactions are labelled (illicit or licit); the rest are
  unknown and were excluded from this stage


## How to run

#### Requirements
-  Python 3.13
- Install dependencies from the project root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Download Datasets from Kaggle
- Download 3 csv files from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Paste 3 csv in data/raw file

#### Run the notebooks in order
Open each notebook from the `notebooks/` folder and use **Run All**. Each stage depends on the one before it.

| Stage | Notebook | What it does | Needs | Produces |
|---|---|---|---|---|
| 1 | `data_exploration.ipynb` | Loads, merges and checks the data, makes the exploration plots | `data/raw/*.csv` | `data/processed/df_labeled.csv` |
| 2 | `baseline_model.ipynb` | XGBoost baseline, feature importance, class-weighted XGBoost | `df_labeled.csv` | Baseline metrics |
| 3 | `gnn_model.ipynb` | GCN and GraphSAGE on the train 1-34 / test 35-49 split | `data/raw/*.csv` | GNN metrics |
| 4 | `model_comparison.ipynb` | MLP, GCN and GraphSAGE over several seeds (train 1-24, val 25-34, test 35-49) | `data/raw/*.csv` | `results/split_1_24.npz` |
| 5 | `xgboost_comparison.ipynb` | XGBoost on the same 1-24 split for a fair comparison | `results/split_1_24.npz` | Final XGBoost row |

Stage 5 must be run after stage 4, because it reads the file stage 4 saves.

#### Project structure
```
money-laundry-detection/
├── data/raw/          # Elliptic CSVs
├── data/processed/    # df_labeled.csv
├── notebooks/         
├── results/           # seed results and saved split
└── src/functions/     # data_loader.py, models.py
```


## 1. XGBoost

### Method
1. Data quality checks: 0 missing values, 0 duplicate rows or transaction IDs, all 165 features numeric (float64). Features appear already standardised, so no scaling was needed for models.
2. Time-based split: trained on time steps 1-34, tested on 35-49. This mimics predicting future transactions from past ones which is more realistic than a random split.
3. Model: XGBoost (100 trees, max depth 6), used default settings, no tuning.
4. Evaluation: precision, recall and F1 on the illicit class.

### Results (test set: 16,670 transactions, 1,083 illicit)

| Model | Illicit precision | Illicit recall | Illicit F1 |
|---|---|---|---|
| XGBoost, all 165 features | 0.91 | 0.72 | 0.81 |
| XGBoost, local features only | 0.85 | 0.70 | 0.77 |
| XGBoost, all features except feature_5 | 0.89 | 0.73 | 0.80 |

Overall accuracy was 0.98, but this is misleading: a model that labelled everything "licit" would score about 93.5% accuracy on this test set while catching no illicit transactions.

Confusion matrix (all features): 

![Confusion matrix](image.png)

### Key findings
- Precision is high (0.91): when the model flags a transaction, it is right about 9 times in 10, so investigators see few false alarms.
- Recall is lower (0.72): roughly 1 in 4 illicit transactions is missed. In a real compliance setting this precision/recall trade-off is the central design decision.
- Neighbour-aggregated features add a modest improvement (F1 0.77 to 0.81, mostly via precision), so surrounding-transaction information carries some signal, but a tabular model already captures most of it.
- feature_5 has the highest importance (~0.5), yet removing it barely changes performance (F1 0.80). Other features carry overlapping information, so importance scores alone should not be read as "this feature is essential".

### Limitations
- Results come from a single run and a single time split, with no hyperparameter tuning or confidence intervals; differences of about 0.01 F1 are within noise.
- Features are anonymised, so findings are reported by feature number and cannot be interpreted in business terms.
- Feature importance describes what the model used, not what causes illicit behaviour.
- Cryptocurrency data is a proxy; real bank transaction data would differ.


## 2. Graph Convolutional Network (GCN)

**Questions**
1. Does network structure improve illicit detection beyond a tabular model?

### Method 
- Graph: all 203,769 transactions as nodes and the payment-flow edges converted to undirected. 
- Features: the same 165 anonymised features used for the baseline. 
- Split: same as the baseline (train on time steps 1-34, test on 35-49) 
- Model: 2-layer GCN, 200 epochs, seed 42. 
- Class imbalance: cross-entropy loss with the illicit class weighted by the licit-to-illicit ratio in the training data 7.63.

### Results

| Model | Illicit precision | Illicit recall | Illicit F1 |
|---|---|---|---|
| XGBoost (all 165 features) | 0.91 | 0.72 | 0.81 |
| GCN (single run, untuned) | 0.62 | 0.63 | 0.63 |

Overall GCN accuracy was 0.95, which again says little on imbalanced data. 

### Interpretation
- The basic GCN could not surpass the tree benchmark: the illicit set F1 score was 0.63 compared to 0.81, mainly due to a difference in precision (0.62 vs. 0.91).
- Potential reasons (not yet tested): neighbour averaging may blur the transaction-level signal, the model may be undertrained, and the neighbour-aggregated features may already provide XGBoost with most of the network information.

### Limitations
- One run, one seed, no validation set, fixed 200 epochs and no hyperparameter tuning. Differences among a few points could be noise. 
- Transductive setup: the entire graph, including features of test period nodes, is known during training, but not the test labels. 
- The fixed 0.5 decision threshold affects precision and recall; threshold-free metrics are reported(PR-AUC). 


## 3. GraphSAGE

Same graph, features, split, class weighting, hyperparameters and seed as the GCN. The only change is the layer type: GraphSAGE keeps a node's own features separate from the aggregate of its neighbours', whereas a GCN layer averages them together.

### Results

| Model | Illicit precision | Illicit recall | Illicit F1 |
|---|---|---|---|
| XGBoost (all 165 features) | 0.91 | 0.72 | 0.81 |
| GCN (single run, untuned) | 0.62 | 0.63 | 0.63 |
| GraphSAGE (single run, untuned) | 0.68 | 0.65 | 0.67 |

### Interpretation
- GraphSAGE scored about 0.04 higher F1 than the GCN, but it is one seed, so this gap may be noise. It still trails XGBoost by a wide margin, again mainly through lower precision (0.68 vs 0.91).
- A possible reason for GraphSAGE's small gain is that it preserves a node's own features.


### Threshold-independent comparison (PR-AUC, test set)

| Model | PR-AUC |
|---|---|
| XGBoost (all features) | 0.801 |
| GraphSAGE (single run) | 0.715 |
| GCN (single run) | 0.591 |

XGBoost ranks illicit transactions better than both GNNs, so the gap is not only an effect of the 0.5 decision threshold or of class weighting. GraphSAGE is closer to XGBoost than the GCN is, but each GNN is a single run.


## Model Comparison
(Train steps 1-24, Validation 25-34, Test 35-49)

The XGBoost model with class weights achieved the best performance (precision 0.855, recall 0.674, F1 0.754, PR-AUC 0.770; single training run, fixed seed). Out of the neural models (mean ± standard deviation from 3 seeds), GraphSAGE obtained a PR-AUC score of 0.453 ± 0.036, better than the MLP that only used features (0.337 ± 0.009) and GCN (0.290 ± 0.023). The GNNs were not hyperparameter optimised. All neural networks exhibited very good recall but poor precision, thereby marking many legal transactions as illegal.

### Interpretation

On this split, adding graph structure helped the neural network only when using GraphSAGE, and no neural model matched XGBoost. A possible reason is that the provided features already encode neighbourhood information, but this was not directly tested.