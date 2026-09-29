# Datasets

Put the files below in `data/raw/`. Loaders match file names with the patterns in `src/config.py` and column
names case-insensitively. The datasets are not redistributed here; please obtain them from the original sources
and respect their terms of use.

| Key | Dataset | Source | Expected file | Rows (churn rate) used in the paper |
|---|---|---|---|---|
| `telco` | IBM Telco Customer Churn | https://www.kaggle.com/datasets/blastchar/telco-customer-churn | `WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 (26.5%) |
| `bank` | Bank Customer Churn ("Churn Modelling") | https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling | `Churn_Modelling.csv` | 10,000 (20.4%) |
| `bankchurners` | Credit Card Customers | https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers | `BankChurners.csv` | 10,127 (16.1%) |
| `iranian` | Iranian Churn (UCI id 563) | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset — doi:10.24432/C5JW3Z | `Customer Churn.csv` | 3,150 (15.7%) |
| `telecom_bigml` | Churn in Telecom's dataset | https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset | `bigml_*.csv` | 3,333 (14.5%) |
| temporal | Online Retail II (UCI id 502) | https://archive.ics.uci.edu/dataset/502/online+retail+ii — doi:10.24432/C5CG6D | `online_retail_II.xlsx` | 1,067,371 transactions → 48,085 customer snapshots |

With the Kaggle CLI (requires an API token):

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
kaggle datasets download -d shrutimechlearn/churn-modelling -p data/raw --unzip
kaggle datasets download -d sakshigoyal7/credit-card-customers -p data/raw --unzip
kaggle datasets download -d becksddf/churn-in-telecoms-dataset -p data/raw --unzip
```

The two UCI datasets can be downloaded as zip files from the pages above and extracted into `data/raw/`.

## Preprocessing decisions that matter

* Identifier columns are removed (`customerID`, `RowNumber`, `CustomerId`, `Surname`, `CLIENTNUM`, `phone number`).
* `BankChurners.csv` contains two columns `Naive_Bayes_Classifier_...` computed from the label. They leak the
  target and are removed by the loader.
* `TotalCharges` in IBM Telco is converted to numeric (blank strings become missing and are imputed in-fold).
* Text columns with more than 50% distinct values are treated as identifiers and dropped.
* Online Retail II: rows without a customer ID are removed; invoices starting with "C" or with negative quantity
  are returns. The cleaned transaction table is cached as `online_retail_II_clean.csv.gz` on first use.
