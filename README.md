# DSLAB Project: Clusters Purity and CTGAN Generated Data

## Overview

This project processes a large, imbalanced financial transactions dataset to prepare it for machine learning experiments. The workflow consists of several main steps:

1. **Data Preprocessing** (`data/pre_processing.ipynb`)
2. **Dual Undersampling** (`data/DUAL_undersampling.ipynb`)
3. **Clustering (Pure Clusters Generation)** (`clusters/clusters_generations.ipynb`)
4. **Cluster Merging (Hybrid Clusters Generation)** (`clusters/clusters_mergeing.ipynb`)
5. **Synthetic Data Generation (CTGAN)** (`ctgan/data_generator_training.ipynb`)
6. **Evaluation and Analysis** (see below)

---

## 1. Data Preprocessing

**Notebook:** `data/pre_processing.ipynb`

### What it does

- Loads the raw dataset from `dataset/dataset.csv`.
- Encodes categorical columns and drops unnecessary columns.
- Splits the data into majority (non-fraud) and minority (fraud) classes.
- Saves the split datasets as:
  - `dataset/majority.csv` (majority class)
  - `dataset/minority.csv` (minority class)

### How to run

Open the notebook and run all cells:

```bash
jupyter notebook data/pre_processing.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute data/pre_processing.ipynb
```

**Input required:**

- `dataset/dataset.csv` (should be present in the `dataset/` folder)

**Output:**

- `dataset/majority.csv`
- `dataset/minority.csv`

---

## 2. Dual Undersampling

**Notebook:** `data/DUAL_undersampling.ipynb`

### What it does

- Loads the preprocessed `majority.csv` and `minority.csv` files.
- Uses active learning (uncertainty sampling) to select the most informative majority class samples.
- Produces a balanced dataset for downstream tasks.
- Saves the selected majority samples as:
  - `dataset/majority_informative.csv`

### How to run

Open the notebook and run all cells:

```bash
jupyter notebook data/DUAL_undersampling.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute data/DUAL_undersampling.ipynb
```

**Input required:**

- `dataset/majority.csv`
- `dataset/minority.csv`

**Output:**

- `dataset/majority_informative.csv`

---

## 3. Clustering (Pure Clusters Generation)

**Notebook:** `clusters/clusters_generations.ipynb`

### What it does

- Loads the balanced datasets (e.g., `minority.csv`, `majority_informative.csv`).
- Applies clustering algorithms to generate pure clusters for both minority and majority classes.
- Saves the resulting clusters as separate CSV files, such as:
  - `dataset/min_0.csv`, `dataset/min_1.csv`, `dataset/min_2.csv` (minority clusters)
  - `dataset/informative_cluster_0.csv`, `dataset/informative_cluster_1.csv` (majority clusters)

### How to run

Open the notebook and run all cells:

```bash
jupyter notebook clusters/clusters_generations.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute clusters/clusters_generations.ipynb
```

**Input required:**

- `dataset/minority.csv`
- `dataset/majority_informative.csv`

**Output:**

- `dataset/min_0.csv`, `dataset/min_1.csv`, `dataset/min_2.csv`
- `dataset/informative_cluster_0.csv`, `dataset/informative_cluster_1.csv`

---

## 4. Cluster Merging (Hybrid Clusters Generation)

**Notebook:** `clusters/clusters_mergeing.ipynb`

### What it does

- Loads the pure cluster files generated in the previous step.
- Merges clusters in various combinations to create hybrid (mixed) clusters for further experiments.
- Saves the merged clusters as new CSV files, such as:
  - `dataset/min_0_informative_cluster_0.csv`, `dataset/min_1_informative_cluster_1.csv`, etc.
  - `dataset/min_0_min_1_informative_cluster_0_informative_cluster_1.csv` (for multi-cluster merges)

### How to run

Open the notebook and run all cells:

```bash
jupyter notebook clusters/clusters_mergeing.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute clusters/clusters_mergeing.ipynb
```

**Input required:**

- `dataset/min_0.csv`, `dataset/min_1.csv`, `dataset/min_2.csv`
- `dataset/informative_cluster_0.csv`, `dataset/informative_cluster_1.csv`

**Output:**

- Various merged cluster files in the `dataset/` directory (see file names above)

---

## 5. Synthetic Data Generation (CTGAN)

**Notebook:** `ctgan/data_generator_training.ipynb`

### What it does

- Loads the merged and pure cluster datasets generated in the previous steps.
- Trains a CTGAN model for each dataset to generate synthetic (fake) data.
- Saves the generated synthetic datasets as CSV files prefixed with `fake_`, such as:
  - `dataset/fake_df_min.csv`, `dataset/fake_df_majority.csv`
  - `dataset/fake_min_0_informative_cluster_0.csv`, `dataset/fake_min_1_informative_cluster_1.csv`, etc.
  - `dataset/fake_min_0_min_1_informative_cluster_0_informative_cluster_1.csv` (for multi-cluster merges)

### How to run

Open the notebook and run all cells:

```bash
jupyter notebook ctgan/data_generator_training.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute ctgan/data_generator_training.ipynb
```

**Input required:**

- Cluster and merged cluster CSV files from the previous steps (see `dataset/` directory)

**Output:**

- Synthetic datasets in the `dataset/` directory, prefixed with `fake_`

---

## 6. Evaluation and Analysis

**Notebooks:**

- `eval/pure_fake_vs_all_fake.ipynb`
- `eval/coorelations.ipynb`
- `eval/majority_vs_minority.ipynb`
- `eval/mixed_clusters.ipynb`
- `eval/everything_mixed.ipynb`
- `eval/highest_purity.ipynb`
- `clusters/clusters_similarity.ipynb`

### What it does

- Evaluates, visualizes, and compares the real and synthetic datasets, cluster quality, and model performance using a variety of metrics and plots.
- Each notebook focuses on a different aspect of the analysis (e.g., cluster similarity, purity, correlations, classification performance).

### How to run

Open each notebook and run all cells, one by one, to perform the full evaluation:

```bash
jupyter notebook eval/pure_fake_vs_all_fake.ipynb
jupyter notebook eval/coorelations.ipynb
jupyter notebook eval/majority_vs_minority.ipynb
jupyter notebook eval/mixed_clusters.ipynb
jupyter notebook eval/everything_mixed.ipynb
jupyter notebook eval/highest_purity.ipynb
jupyter notebook clusters/clusters_similarity.ipynb
```

Or, from the command line:

```bash
jupyter nbconvert --to notebook --execute eval/pure_fake_vs_all_fake.ipynb
jupyter nbconvert --to notebook --execute eval/coorelations.ipynb
jupyter nbconvert --to notebook --execute eval/majority_vs_minority.ipynb
jupyter nbconvert --to notebook --execute eval/mixed_clusters.ipynb
jupyter nbconvert --to notebook --execute eval/everything_mixed.ipynb
jupyter nbconvert --to notebook --execute eval/highest_purity.ipynb
jupyter nbconvert --to notebook --execute clusters/clusters_similarity.ipynb
```

**Note:**

- Run the evaluation notebooks in any order, but for a comprehensive analysis, it is recommended to run all of them.
- Outputs include plots, tables, and metrics for comparison and reporting.

---

## Dependencies

Install the following Python packages (preferably in a virtual environment):

```
pandas
numpy
scikit-learn
modAL
torch
matplotlib
seaborn
kagglehub
jupyter
sdv
```

You can install them with:

```bash
pip install pandas numpy scikit-learn modAL torch matplotlib seaborn kagglehub jupyter sdv
```

---

## Notes

- The raw dataset is large (~322MB). Ensure you have enough memory to process it.
- The output files will be saved in the `dataset/` directory.
- For any issues, check the notebook outputs for error messages or missing dependencies.
