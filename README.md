# Thesis Publication

## 👂 Repository Structure

The repository contains experiments on multiple telecom datasets using different **Graph Neural Network (GNN) architectures**:

```
thesis-publication/
│── MV/        # Mobile Vikings dataset experiments
│── PRE/       # Proximus Prepaid dataset experiments
│── POST/      # Proximus Postpaid dataset experiments
│── requirements.txt
```

Inside each dataset folder, experiments are organized by **GNN architecture**:

```
MV/
│── GAT/        # Graph Attention Network experiments
│── GCN/        # Graph Convolutional Network experiments
│── GIN/        # Graph Isomorphism Network experiments
│── GRAPHSAGE/  # GraphSAGE experiments
```

The same structure applies to `PRE/` and `POST/` dataset folders.

Each GNN folder contains:

* Model code and training scripts
* Evaluation results
* Logs of experiments

---

## 📈 Evaluation Metrics

Model performance is evaluated using **machine learning metrics** and **business-oriented metrics**.

### Machine Learning Metrics

* **Loss**: Optimization objective value (lower is better)
* **AUC (Area Under ROC Curve)**: Measures ability to rank churners vs. non-churners (0.5 = random, 1.0 = perfect)
* **AUPRC (Area Under Precision-Recall Curve)**: Focuses on precision for the positive churn class, especially important in imbalanced datasets

### Lift Metrics

Lift measures how well the model identifies churners in the top fraction of ranked predictions:

| Metric     | Fraction of top predictions | 
| ---------- | --------------------------- | 
| lift\_0005 | 0.05%                       | 
| lift\_001  | 0.1%                        |
| lift\_005  | 0.5%                        |
| lift\_01   | 1%                          |

*(A lift of 1.0 corresponds to random selection.)*

### Business-Oriented Metrics

Implemented using the [EMP-PY package](https://pypi.org/project/EMP-PY/):

* **EMP (Expected Maximum Profit)**: Maximum profit achievable with an optimal retention campaign
* **MP (Maximum Profit)**: Actual maximum profit based on the decision threshold that maximizes profit
