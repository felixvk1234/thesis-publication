import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
from datetime import datetime
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from EMP.metrics import empCreditScoring, empChurn
from copy import deepcopy

print("[Checkpoint] All libraries imported successfully.")

# Configuration
class Config:
    DATA_DIR = r"/data/leuven/373/vsc37331/Mobile_Vikings/"
    TRAIN_EDGE = "SN_M1t2_c.csv"
    TRAIN_LABEL = "L_M3.csv"
    TRAIN_RMF = "train_rmf_LT.csv"
    VAL_EDGE = "SN_M2t3_c.csv"
    VAL_LABEL = "L_M4.csv"
    VAL_RMF = "val_rmf_LT.csv"
    TEST_EDGE = "SN_M3t4_c.csv"
    TEST_LABEL = "L_test.csv"
    TEST_RMF = "test_rmf_LT.csv"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LEARNING_RATES = [0.01, 0.0001]
    HIDDEN_CHANNELS = [32, 256]
    LAYERS = [1, 3]
    PATIENCE = 15
    MAX_EPOCHS = 200
    DROPOUT_RATE = 0.5
    EMBEDDING_DIM = 32
    # Focal Loss hyperparameters to tune
    FOCAL_ALPHAS = [0.5, 0.75]   # Prioritize positive class more
    FOCAL_GAMMAS = [1.0, 2.0]    # Focus on hard examples more
    CLIP_GRAD_NORM = 1.0
    # GAT-specific hyperparameters
    HEADS = [4, 8]               # Number of attention heads
    CONCAT = [True, False]       # Whether to concatenate or average multi-head attention outputs


print(f"[Checkpoint] Configuration initialized")
print(f"[Checkpoint] Using device: {Config.DEVICE}")

os.chdir(Config.DATA_DIR)
print(f"[Checkpoint] Working directory set to: {os.getcwd()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply torch.clamp to prevent numerical instability
        inputs = torch.clamp(inputs, min=-88, max=88)
        
        # Binary cross-entropy term
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities safely
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, min=1e-7, max=1-1e-7)  # Avoid 0 and 1 for numerical stability
        
        # Focal Loss modulating factor
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final loss
        focal_loss = alpha_weight * modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class GraphDataProcessor:
    @staticmethod
    def load_churned_users():
        """Load the list of users who churned in month 1"""
        try:
            churn_m1 = pd.read_csv("L_M1.csv")
            nodes_to_remove = churn_m1[churn_m1['churn_m1'] == 1]['USR'].values.astype('int64')
            churner_set_m1 = set(nodes_to_remove)
            print(f"[Checkpoint] Identified {len(churner_set_m1)} users who churned in month 1")
            return churner_set_m1
        except FileNotFoundError:
            print("[Checkpoint] WARNING: L_M1.csv not found. No users will be excluded.")
            return set()
        except Exception as e:
            print(f"[Checkpoint] ERROR loading churned users: {str(e)}")
            return set()

    @staticmethod
    def remove_nodes_and_create_data(rmf_path, edge_path, label_path, churner_set_m1):
        node_df = pd.read_csv(rmf_path)
        edge_df = pd.read_csv(edge_path)
        label_df = pd.read_csv(label_path)
        print(f"[Checkpoint] Loaded RMF, edge, and label data")

        print(f"[Checkpoint] Starting remove_nodes_and_create_data")
        
        # Step 1: Map original node indices (1-based) to USR
        usr_list_old = node_df['USR'].tolist()
        index_to_usr = {idx: usr for idx, usr in enumerate(usr_list_old, start=1)}
        print(f"[Checkpoint] Mapped {len(index_to_usr)} node indices to USRs")

        # Step 2: No longer filtering out first month churners
        # We keep all users in the node and label data
        print(f"[Checkpoint] Keeping all users including first month churners")

        # Step 3: Map edge list from index to USR
        edge_df['i'] = edge_df['i'].map(index_to_usr)
        edge_df['j'] = edge_df['j'].map(index_to_usr)

        # Step 4: No longer filtering out edges with churners
        print(f"[Checkpoint] Keeping all edges including those involving first month churners")

        # Step 5: Map remaining USRs to new 0-based indices
        usr_list_new = node_df['USR'].tolist()
        usr_to_index = {usr: idx for idx, usr in enumerate(usr_list_new)}
        print(f"[Checkpoint] Created mapping for {len(usr_to_index)} users")

        # Step 6: Convert edge list from USR to index and extract edge weights
        mapped_i = edge_df['i'].map(usr_to_index)
        mapped_j = edge_df['j'].map(usr_to_index)
        
        # No longer extracting edge weights
        print(f"[Checkpoint] Converting edge list to indices")

        missing_i = mapped_i.isna().sum()
        missing_j = mapped_j.isna().sum()
        print(f"[Checkpoint] Missing i mappings: {missing_i}, Missing j mappings: {missing_j}")

        # Filter out any edges with NA mappings
        valid_edges = ~(mapped_i.isna() | mapped_j.isna())
        mapped_i = mapped_i[valid_edges].values
        mapped_j = mapped_j[valid_edges].values

        # Create edge indices for undirected graph without edge weights
        edge_index_0 = torch.tensor([mapped_i, mapped_j], dtype=torch.long)
        edge_index_1 = torch.tensor([mapped_j, mapped_i], dtype=torch.long)
        edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
        
        print(f"[Checkpoint] Created undirected edge index with shape: {edge_index.shape}")

        # Step 7: Prepare node features
        feature_df = node_df.drop(columns=['USR', 'churn'], errors='ignore')
        # Modified to remove features with '30' and '90' instead of '60' and '90'
        feature_df = feature_df.drop(columns=[col for col in feature_df.columns if '30' in col or '90' in col])
        print(f"[Checkpoint] Remaining feature columns: {list(feature_df.columns)}")

        scaler = StandardScaler()
        x = torch.tensor(scaler.fit_transform(feature_df), dtype=torch.float)
        print(f"[Checkpoint] Node feature tensor shape: {x.shape}")

        # Step 8: Prepare label tensor
        y = torch.tensor(label_df.iloc[:, -1].values, dtype=torch.long)
        print(f"[Checkpoint] Label tensor shape: {y.shape}")
        print(f"[Checkpoint] Label distribution: {torch.bincount(y)}")

        # Step 9: Create and return PyG Data object without edge weights
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            num_nodes=x.shape[0], 
            num_edges=edge_index.shape[1], 
            num_features=x.shape[1]
        )
        print(f"[Checkpoint] Created Data object with {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")

        return data

    @staticmethod
    def get_class_distribution(data):
        """Returns class distribution for monitoring"""
        y = data.y
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        print(f"[Checkpoint] Class distribution - Positives: {num_pos}, Negatives: {num_neg}")
        return num_pos, num_neg

class GAT(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_nodes, num_layers, heads=8, concat=True, dropout_rate=Config.DROPOUT_RATE):
        """
        Graph Attention Network (GAT) model with BatchNorm, ELU activation, and dropout.

        Args:
            input_dim (int): Dimension of input features
            embedding_dim (int): Dimension of node embeddings
            hidden_dim (int): Dimension of hidden layers
            num_nodes (int): Total number of nodes in the graph
            num_layers (int): Number of GAT layers
            heads (int): Number of attention heads
            concat (bool): Whether to concatenate or average multi-head attention outputs
            dropout_rate (float): Dropout probability
        """
        super().__init__()
        print(f"[Checkpoint] Initializing GAT with {input_dim} input dim, {embedding_dim} embedding dim, "
              f"{hidden_dim} hidden dim, {num_layers} layers, {heads} heads, concat={concat}")

        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat = concat
        self.heads = heads

        # Node embedding
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)

        # Feature transformation
        self.feature_transform = nn.Linear(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.feature_transform.weight, gain=0.1)

        # Combine embedding + features
        self.combine = nn.Linear(embedding_dim * 2, hidden_dim)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)

        # GAT layers and BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # Input and output dimensions for GAT layers depend on heads and concat option
            if i == 0:
                in_channels = hidden_dim
            else:
                in_channels = hidden_dim * heads if concat else hidden_dim
                
            # For the final layer, adjust output dimension
            if i == num_layers - 1:
                out_channels = hidden_dim // heads if concat else hidden_dim
            else:
                out_channels = hidden_dim
                
            conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout_rate,
                add_self_loops=True
            )
            self.convs.append(conv)
            
            # BatchNorm dimension depends on whether we concatenate or average heads
            bn_dim = out_channels * heads if concat else out_channels
            self.bns.append(BatchNorm(bn_dim))

        # Final linear output layer
        final_dim = hidden_dim * heads if concat else hidden_dim
        self.lin = nn.Linear(final_dim, 1)
        nn.init.xavier_uniform_(self.lin.weight, gain=0.1)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Checkpoint] Total parameters: {total_params}")
    
    def forward(self, x, edge_index):
        # Node ID embeddings
        node_indices = torch.arange(x.size(0), device=x.device)
        node_emb = self.embedding(node_indices)

        # Feature embedding
        feature_emb = self.feature_transform(x)

        # Combine both
        h = torch.cat([node_emb, feature_emb], dim=1)
        h = F.relu(self.combine(h))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GAT layers with residual connections when possible
        prev_h = h
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            
            # Add residual connection when dimensions match
            if prev_h.shape == h.shape:
                h = h + prev_h
            prev_h = h
            
            h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # Final prediction
        out = self.lin(h)
        return torch.clamp(out, min=-10, max=10)  # Prevent extreme values

class GATTrainer:
    def __init__(self, model, device='cpu', alpha=0.25, gamma=2.0):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        
    def train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.float().to(self.device)
        
        # Call model
        out = self.model(x, edge_index)
        loss = self.criterion(out.squeeze(), y)
        
        if not torch.isfinite(loss).all():
            print("[Checkpoint] WARNING: Non-finite loss detected, skipping backward pass")
            return float('nan')
            
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD_NORM)
        
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, calculate_emp=False):
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.float().to(self.device)
            
            try:
                # Call model
                out = self.model(x, edge_index)
                loss = self.criterion(out.squeeze(), y)

                # Handle potential NaN/Inf values
                probs = torch.sigmoid(out).squeeze().cpu().numpy()
                
                # Check for and handle NaN values
                if np.isnan(probs).any():
                    print("[Checkpoint] WARNING: NaN detected in probabilities, replacing with 0.5")
                    probs = np.nan_to_num(probs, nan=0.5)
                
                y_true = data.y.cpu().numpy()

                # Calculate AUC
                auc_roc = roc_auc_score(y_true, probs)
                
                # Calculate AUPRC
                precision, recall, _ = precision_recall_curve(y_true, probs)
                auprc = auc(recall, precision)
                
                # Calculate various lift metrics
                lift_0005 = self.calculate_lift(y_true, probs, 0.005)
                lift_001 = self.calculate_lift(y_true, probs, 0.01)
                lift_005 = self.calculate_lift(y_true, probs, 0.05)
                lift_01 = self.calculate_lift(y_true, probs, 0.1)

                metrics = {
                    'loss': loss.item(),
                    'auc': auc_roc,
                    'auprc': auprc,
                    'lift_0005': lift_0005,
                    'lift_001': lift_001,
                    'lift_005': lift_005,
                    'lift_01': lift_01,
                    'probs': probs,
                    'labels': y_true
                }
                
                if calculate_emp:
                    try:
                        emp_output = empChurn(probs, y_true, return_output=True, print_output=False)
                        metrics['emp'] = float(emp_output.EMP)
                        metrics['mp'] = float(emp_output.MP)
                    except Exception as e:
                        print(f"[Checkpoint] Error calculating EMP metrics: {str(e)}")
                        metrics['emp'] = 0.0
                        metrics['mp'] = 0.0
                else:
                    metrics['emp'] = 0.0
                    metrics['mp'] = 0.0
                
                return metrics
            except Exception as e:
                print(f"[Checkpoint] Error during evaluation: {str(e)}")
                # Return default metrics in case of error
                return {
                    'loss': float('inf'),
                    'auc': 0.5,
                    'auprc': 0.5,
                    'lift_0005': 1.0,
                    'lift_001': 1.0,
                    'lift_005': 1.0,
                    'lift_01': 1.0,
                    'emp': 0.0,
                    'mp': 0.0,
                    'probs': np.array([0.5]),
                    'labels': np.array([0])
                }

    @staticmethod
    def calculate_lift(y_true, y_prob, percentage):
        try:
            sorted_indices = np.argsort(y_prob)[::-1]
            n_top = max(1, int(len(y_true) * percentage))
            top_indices = sorted_indices[:n_top]
            top_positive = y_true[top_indices].sum()
            top_positive_rate = top_positive / n_top
            
            overall_positive = y_true.sum()
            overall_positive_rate = overall_positive / len(y_true)
            
            if overall_positive_rate == 0:
                return 1.0
            
            return top_positive_rate / overall_positive_rate
        except Exception as e:
            print(f"[Checkpoint] Error calculating lift: {str(e)}")
            return 1.0
        
class Experiment:
    def __init__(self):
        print("[Checkpoint] ====== Starting Experiment setup ======")
        
        self.churned_users = GraphDataProcessor.load_churned_users()

        print("[Checkpoint] Loading training data with RMF features")
        self.data_train = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.TRAIN_EDGE, 
            label_path=Config.TRAIN_LABEL,
            rmf_path=Config.TRAIN_RMF,
            churner_set_m1=self.churned_users
        )
        GraphDataProcessor.get_class_distribution(self.data_train)
        
        print("[Checkpoint] Loading validation data with RMF features")
        self.data_val = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.VAL_EDGE, 
            label_path=Config.VAL_LABEL,
            rmf_path=Config.VAL_RMF,
            churner_set_m1=self.churned_users
        )

        print("[Checkpoint] Loading test data with RMF features")
        self.data_test = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.TEST_EDGE, 
            label_path=Config.TEST_LABEL,
            rmf_path=Config.TEST_RMF,
            churner_set_m1=self.churned_users
        )
        
        print("[Checkpoint] Experiment initialization complete")

    def run_hyperparameter_tuning(self):
        print("[Checkpoint] ====== Starting Hyperparameter Tuning ======")
        
        # Only track best model by AUPRC now
        best_val_auprc = 0
        best_model = None
        best_config = None
        
        num_features = self.data_train.x.shape[1]
        print(f"[Checkpoint] Number of input features: {num_features}")

        total_configs = len(Config.LEARNING_RATES) * len(Config.HIDDEN_CHANNELS) * len(Config.LAYERS) * \
                        len(Config.FOCAL_ALPHAS) * len(Config.FOCAL_GAMMAS) * len(Config.HEADS) * len(Config.CONCAT)
        print(f"[Checkpoint] Testing {total_configs} configurations")
        
        config_num = 1
        
        for lr in Config.LEARNING_RATES:
            for hidden in Config.HIDDEN_CHANNELS:
                for num_layers in Config.LAYERS:
                    for alpha in Config.FOCAL_ALPHAS:
                        for gamma in Config.FOCAL_GAMMAS:
                            for heads in Config.HEADS:
                                for concat in Config.CONCAT:
                                    print(f"\n[Checkpoint] ====== Configuration {config_num}/{total_configs} ======")
                                    print(f"[Checkpoint] Training with lr={lr}, hidden={hidden}, layers={num_layers}, "
                                          f"alpha={alpha}, gamma={gamma}, heads={heads}, concat={concat}")
                                    config_num += 1

                                    try:
                                        model = GAT(
                                            input_dim=num_features,
                                            embedding_dim=Config.EMBEDDING_DIM,
                                            hidden_dim=hidden, 
                                            num_layers=num_layers,
                                            num_nodes=self.data_train.num_nodes,
                                            heads=heads,
                                            concat=concat,
                                            dropout_rate=Config.DROPOUT_RATE
                                        )
                                        
                                        trainer = GATTrainer(model, Config.DEVICE, alpha=alpha, gamma=gamma)
                                        
                                        # Use Adam with weight decay to prevent overfitting and improve stability
                                        optimizer = torch.optim.Adam(
                                            model.parameters(), 
                                            lr=lr,
                                            weight_decay=1e-5  # Add weight decay
                                        )

                                        current_best_val_auprc = 0
                                        epochs_no_improve = 0
                                        best_epoch = 0
                                        nan_epochs = 0  # Count consecutive NaN epochs

                                        print(f"[Checkpoint] Starting training for up to {Config.MAX_EPOCHS} epochs")
                                        
                                        for epoch in range(1, Config.MAX_EPOCHS + 1):
                                            print(f"\n[Checkpoint] --- Epoch {epoch}/{Config.MAX_EPOCHS} ---")
                                            
                                            loss = trainer.train(self.data_train, optimizer)
                                            
                                            if np.isnan(loss):
                                                nan_epochs += 1
                                                print(f"[Checkpoint] NaN loss detected, nan_epochs={nan_epochs}")
                                                if nan_epochs >= 3:  # Skip this configuration after 3 consecutive NaN epochs
                                                    print("[Checkpoint] Too many NaN epochs, skipping configuration")
                                                    break
                                            else:
                                                nan_epochs = 0  # Reset counter on successful epoch
                                                print(f"[Checkpoint] Epoch {epoch} training loss: {loss:.6f}")

                                            if epoch % 5 == 0 and not np.isnan(loss):
                                                val_metrics = trainer.evaluate(self.data_val)
                                                print(f"[Checkpoint] Validation metrics - AUC: {val_metrics['auc']:.6f}, AUPRC: {val_metrics['auprc']:.6f}, Loss: {val_metrics['loss']:.6f}")
                                                print(f"[Checkpoint] Validation lifts - @0.5%: {val_metrics['lift_0005']:.6f}, @1%: {val_metrics['lift_001']:.6f}, @5%: {val_metrics['lift_005']:.6f}, @10%: {val_metrics['lift_01']:.6f}")

                                                # Update best model based on AUPRC improvement
                                                if val_metrics['auprc'] > best_val_auprc:
                                                    print(f"[Checkpoint] New best model found! AUPRC: {val_metrics['auprc']:.6f} (previous best: {best_val_auprc:.6f})")
                                                    best_val_auprc = val_metrics['auprc']
                                                    best_model = deepcopy(model)
                                                    best_config = (lr, hidden, num_layers, alpha, gamma, heads, concat)
                                                
                                                # Early stopping based on AUPRC
                                                if val_metrics['auprc'] > current_best_val_auprc:
                                                    current_best_val_auprc = val_metrics['auprc']
                                                    best_epoch = epoch
                                                    epochs_no_improve = 0
                                                else:
                                                    epochs_no_improve += 1
                                                    if epochs_no_improve >= Config.PATIENCE:
                                                        print(f"[Checkpoint] Early stopping at epoch {epoch} (no AUPRC improvement for {Config.PATIENCE} evaluations)")
                                                        break
                                        
                                        print(f"[Checkpoint] Configuration completed. Best AUPRC: {current_best_val_auprc:.6f}")
                                    except Exception as e:
                                        print(f"[Checkpoint] Error during configuration {config_num-1}: {str(e)}")
                                        print("[Checkpoint] Skipping to next configuration")
                                        continue

        print("\n[Checkpoint] ====== Final Evaluation ======")
        
        # Evaluate the best model
        if best_model is not None:
            print(f"[Checkpoint] Evaluating best model")
            _, _, _, best_alpha, best_gamma, _, _ = best_config
            test_metrics = GATTrainer(best_model, Config.DEVICE, best_alpha, best_gamma).evaluate(self.data_test, calculate_emp=True)
            
            print(f"[Checkpoint] Best config (lr={best_config[0]}, hidden={best_config[1]}, layers={best_config[2]}, alpha={best_config[3]}, gamma={best_config[4]}, heads={best_config[5]}, concat={best_config[6]}):")
            print(f"[Checkpoint] Test AUPRC={test_metrics['auprc']:.6f}")
            print(f"[Checkpoint] Test AUC={test_metrics['auc']:.6f}")
            print(f"[Checkpoint] Test EMP={test_metrics['emp']:.6f}")
            print(f"[Checkpoint] Test MP={test_metrics['mp']:.6f}")
            print(f"[Checkpoint] Test Lift@0.5%={test_metrics['lift_0005']:.6f}")
            print(f"[Checkpoint] Test Lift@1%={test_metrics['lift_001']:.6f}")
            print(f"[Checkpoint] Test Lift@5%={test_metrics['lift_005']:.6f}")
            print(f"[Checkpoint] Test Lift@10%={test_metrics['lift_01']:.6f}")

            # Save model predictions for further analysis
            try:
                print("\n[Checkpoint] Saving best model predictions for analysis")
                
                # Create DataFrame with predictions
                predictions_df = pd.DataFrame({
                    'true_labels': test_metrics['labels'],
                    'predicted_probs': test_metrics['probs']
                })
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                predictions_df.to_csv(f"gat_best_model_predictions_{timestamp}.csv", index=False)
                print(f"[Checkpoint] Predictions saved as gat_best_model_predictions_{timestamp}.csv")
            except Exception as e:
                print(f"[Checkpoint] Error saving predictions: {str(e)}")

        return best_model

if __name__ == "__main__":
    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("[Checkpoint] ====== Script Started ======")
    try:
        experiment = Experiment()
        best_model = experiment.run_hyperparameter_tuning()
        print("[Checkpoint] ====== Script Finished ======")
    except Exception as e:
        print(f"[Checkpoint] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()