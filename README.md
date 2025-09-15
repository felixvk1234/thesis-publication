# Thesis Publication

## 📂 Repository Structure

thesis-publication/
│── MV/ # Experiments on the Mobile Vikings dataset
│── PRE/ # Experiments on the Proximus Prepaid dataset
│── POST/ # Experiments on the Proximus Postpaid dataset
│── requirements.txt

Inside each dataset folder, experiments are organized by **GNN architecture**:

MV/
│── GAT/ # Graph Attention Network experiments
│── GCN/ # Graph Convolutional Network experiments
│── GIN/ # Graph Isomorphism Network experiments
│── GRAPHSAGE/ # GraphSAGE experiments

## 📈 Evaluation Metrics

Model performance is evaluated using both **machine learning metrics** and **business-oriented metrics**.

- **Loss**: Optimization objective value (lower is better).  
- **AUC (Area Under ROC Curve)**: Measures the ability to rank churners vs. non-churners (0.5 = random, 1.0 = perfect).  
- **AUPRC (Area Under Precision-Recall Curve)**: Focuses on the model’s precision for the positive churn class, especially important in imbalanced datasets.  

### Lift Metrics
Lift measures how well the model identifies churners in the top fraction of ranked predictions:  
- **lift_0005** → Top 0.05%  
- **lift_001** → Top 0.1%  
- **lift_005** → Top 0.5%  
- **lift_01** → Top 1%  
(A lift of 1.0 = random selection, higher values indicate better targeting.)

### Business-Oriented Metrics
Implemented using the [EMP-PY package](https://pypi.org/project/EMP-PY/):  
- **EMP (Expected Maximum Profit)**: Estimates the maximum profit achievable when applying an optimal retention campaign.  
- **MP (Maximum Profit)**: The actual maximum profit value based on the decision threshold that maximizes profit.  
