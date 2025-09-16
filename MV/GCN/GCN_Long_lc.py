from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import wandb
import traceback
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from EMP.metrics import empCreditScoring, empChurn
from copy import deepcopy

print("[Checkpoint] All libraries imported successfully.")

# WandB Sweep Configuration
SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {
        'name': 'val_auc',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'hidden_channels': {
            'values': [32, 128, 256]
        },
        'num_layers': {
            'values': [1, 3]
        },
        'focal_alpha': {
            'values': [0.5, 0.75]
        },
        'focal_gamma': {
            'values': [1.0, 2.0]
        }
    }
}

# Configuration
class Config:
    DATA_DIR = r"/data/leuven/373/vsc37331/Mobile_Vikings/"
    # Call count edge files
    TRAIN_EDGE_C = "SN_M1t2_c.csv"
    VAL_EDGE_C = "SN_M2t3_c.csv"
    TEST_EDGE_C = "SN_M3t4_c.csv"
    # Call duration edge files
    TRAIN_EDGE_L = "SN_M1t2_l.csv"
    VAL_EDGE_L = "SN_M2t3_l.csv"
    TEST_EDGE_L = "SN_M3t4_l.csv"
    # Other data files
    TRAIN_LABEL = "L_M3.csv"
    TRAIN_RMF = "train_rmf_LT.csv"
    VAL_LABEL = "L_M4.csv"
    VAL_RMF = "val_rmf_LT.csv"
    TEST_LABEL = "L_test.csv"
    TEST_RMF = "test_rmf_LT.csv"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    HIDDEN_CHANNELS = [32, 128, 256]
    LAYERS = [1, 3]
    PATIENCE = 15
    MAX_EPOCHS = 200
    DROPOUT_RATE = 0.5
    EMBEDDING_DIM = 32
    # Focal Loss hyperparameters to tune
    FOCAL_ALPHAS = [0.5, 0.75]   # Prioritize positive class more
    FOCAL_GAMMAS = [1.0, 2.0]    # Focus on hard examples more
    CLIP_GRAD_NORM = 1.0


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
    def remove_nodes_and_create_data(rmf_path, edge_path_c, edge_path_l, label_path, churner_set_m1):
        """Create PyG data object with dual edge weights (call count and call duration)"""
        node_df = pd.read_csv(rmf_path)
        # Load both edge files - one for call count and one for duration
        edge_df_c = pd.read_csv(edge_path_c)
        edge_df_l = pd.read_csv(edge_path_l)
        label_df = pd.read_csv(label_path)
        print(f"[Checkpoint] Loaded RMF, both edge types (c and l), and label data")

        print(f"[Checkpoint] Starting remove_nodes_and_create_data")
        
        # Step 1: Map original node indices (1-based) to USR
        usr_list_old = node_df['USR'].tolist()
        index_to_usr = {idx: usr for idx, usr in enumerate(usr_list_old, start=1)}
        print(f"[Checkpoint] Mapped {len(index_to_usr)} node indices to USRs")

        # Step 2: No longer filtering out first month churners
        print(f"[Checkpoint] Keeping all users including first month churners")

        # Step 3: Map edge lists from index to USR for both edge types
        edge_df_c['i'] = edge_df_c['i'].map(index_to_usr)
        edge_df_c['j'] = edge_df_c['j'].map(index_to_usr)
        edge_df_l['i'] = edge_df_l['i'].map(index_to_usr)
        edge_df_l['j'] = edge_df_l['j'].map(index_to_usr)

        # Step 4: No longer filtering out edges with churners
        print(f"[Checkpoint] Keeping all edges including those involving first month churners")

        # Step 5: Map remaining USRs to new 0-based indices
        usr_list_new = node_df['USR'].tolist()
        usr_to_index = {usr: idx for idx, usr in enumerate(usr_list_new)}
        print(f"[Checkpoint] Created mapping for {len(usr_to_index)} users")

        # Step 6: Convert edge lists from USR to index and extract edge weights
        # For call count (c) edges
        mapped_i_c = edge_df_c['i'].map(usr_to_index)
        mapped_j_c = edge_df_c['j'].map(usr_to_index)
        edge_weights_c = edge_df_c['x'].values if 'x' in edge_df_c.columns else np.ones(len(mapped_i_c))
        
        # For call duration (l) edges
        mapped_i_l = edge_df_l['i'].map(usr_to_index)
        mapped_j_l = edge_df_l['j'].map(usr_to_index)
        edge_weights_l = edge_df_l['x'].values if 'x' in edge_df_l.columns else np.ones(len(mapped_i_l))
        
        print(f"[Checkpoint] Extracted call count weights with min={edge_weights_c.min()}, max={edge_weights_c.max()}")
        print(f"[Checkpoint] Extracted call duration weights with min={edge_weights_l.min()}, max={edge_weights_l.max()}")

        # Filter out any edges with NA mappings in both edge types
        valid_edges_c = ~(mapped_i_c.isna() | mapped_j_c.isna())
        mapped_i_c = mapped_i_c[valid_edges_c].values
        mapped_j_c = mapped_j_c[valid_edges_c].values
        edge_weights_c = edge_weights_c[valid_edges_c]
        
        valid_edges_l = ~(mapped_i_l.isna() | mapped_j_l.isna())
        mapped_i_l = mapped_i_l[valid_edges_l].values
        mapped_j_l = mapped_j_l[valid_edges_l].values
        edge_weights_l = edge_weights_l[valid_edges_l]

        # Merge edges from both edge types
        # First, create a dictionary with (i, j) as keys
        edge_dict = {}
        
        # Add call count edges
        for idx in range(len(mapped_i_c)):
            i, j = int(mapped_i_c[idx]), int(mapped_j_c[idx])
            count = float(edge_weights_c[idx])
            edge_dict[(i, j)] = [count, 0.0]  # [count, duration]
        
        # Add or update with call duration edges
        for idx in range(len(mapped_i_l)):
            i, j = int(mapped_i_l[idx]), int(mapped_j_l[idx])
            duration = float(edge_weights_l[idx])
            if (i, j) in edge_dict:
                edge_dict[(i, j)][1] = duration
            else:
                edge_dict[(i, j)] = [0.0, duration]
        
        # Convert merged edges to tensors
        edge_pairs = list(edge_dict.keys())
        edge_features = list(edge_dict.values())
        
        src_nodes = [pair[0] for pair in edge_pairs]
        dst_nodes = [pair[1] for pair in edge_pairs]
        
        # Create undirected graph by adding reverse edges
        edge_index_0 = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_index_1 = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
        edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
        
        # Create 2D edge attributes (count, duration)
        edge_attr_0 = torch.tensor(edge_features, dtype=torch.float)
        # Duplicate the edge features for the reverse edges
        edge_attr_1 = torch.tensor(edge_features, dtype=torch.float)
        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=0)
        
        print(f"[Checkpoint] Created undirected edge index with shape: {edge_index.shape}")
        print(f"[Checkpoint] Created edge attributes with shape: {edge_attr.shape}")

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

        # Step 9: Create and return PyG Data object with edge weights
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr,  # 2D edge attributes [count, duration]
            y=y, 
            num_nodes=x.shape[0], 
            num_edges=edge_index.shape[1], 
            num_features=x.shape[1]
        )
        print(f"[Checkpoint] Created Data object with {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features, and 2D edge attributes")

        return data

    @staticmethod
    def get_class_distribution(data):
        """Returns class distribution for monitoring"""
        y = data.y
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        print(f"[Checkpoint] Class distribution - Positives: {num_pos}, Negatives: {num_neg}")
        return num_pos, num_neg

class EnhancedGCN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_channels, num_layers, num_nodes, dropout_rate=Config.DROPOUT_RATE):
        super().__init__()
        print(f"[Checkpoint] Initializing EnhancedGCN with {input_dim} input features, {embedding_dim} embedding dim, {hidden_channels} hidden channels, {num_layers} layers")
        
        # Initialize weights with smaller values to prevent exploding gradients
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
        
        self.feature_transform = nn.Linear(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.feature_transform.weight, gain=0.1)
        
        # Add edge feature transformation layer for 2D edge features
        self.edge_transform = nn.Linear(2, 1)  # Transform [count, duration] to a single weight
        nn.init.xavier_uniform_(self.edge_transform.weight, gain=0.1)
        
        self.combine = nn.Linear(embedding_dim * 2, hidden_channels)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            # Initialize GCN layers with appropriate initialization
            nn.init.xavier_uniform_(conv.lin.weight, gain=0.1)
            self.convs.append(conv)
            
        self.lin = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.lin.weight, gain=0.1)
        
        self.dropout_rate = dropout_rate
        
        # Add BatchNorm to help with training stability
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Checkpoint] Total parameters: {total_params}")

    def forward(self, x, edge_index, edge_attr=None):
        # Input validation
        if torch.isnan(x).any():
            print("[Checkpoint] WARNING: NaN detected in input features")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Process edge attributes if provided
        edge_weight = None
        if edge_attr is not None:
            # Combine call count and duration into a single edge weight
            # We can do this using our edge_transform layer or another method
            edge_weight = self.edge_transform(edge_attr).squeeze(-1)
            # Normalize edge weights to prevent numerical issues
            if edge_weight.max() > 1000:
                edge_weight = edge_weight / edge_weight.max()
            
        node_indices = torch.arange(x.size(0), device=x.device)
        node_emb = self.embedding(node_indices)
        feature_emb = self.feature_transform(x)
        
        combined = torch.cat([node_emb, feature_emb], dim=1)
        h = F.relu(self.combine(combined))  # Using ReLU for more stability than ELU
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        for conv in self.convs:
            # GCNConv in PyG has edge_weight as a named parameter, not a positional one
            h_new = conv(h, edge_index, edge_weight)
            h_new = F.relu(h_new)  # Using ReLU for stability
            h_new = self.batch_norm(h_new)  # Apply batch normalization
            h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)
            
            # Add residual connection for better gradient flow
            if h.shape == h_new.shape:
                h = h_new + h
            else:
                h = h_new
            
        # Ensure output doesn't have extreme values
        out = self.lin(h)
        return torch.clamp(out, min=-10, max=10)  # Prevent extreme values

class GCNTrainer:
    def __init__(self, model, device=None, alpha=0.25, gamma=2.0):
        if device is None:
            device = 'cpu'
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        self.model = model.to(device)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

    def train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.float().to(self.device)
        
        # Handle edge attributes
        edge_attr = None
        if hasattr(data, 'edge_attr'):
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.to(self.device)
        
        out = self.model(x, edge_index, edge_attr)
        loss = self.criterion(out.squeeze(), y)
        
        if not torch.isfinite(loss).all():
            print("[Checkpoint] WARNING: Non-finite loss detected, skipping backward pass")
            return float('nan')
            
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD_NORM)
        
        optimizer.step()
        
        return loss.item()

    def evaluate(self, data, calculate_emp=False):
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.float().to(self.device)
            
            # Handle edge attributes
            edge_attr = None
            if hasattr(data, 'edge_attr'):
                if data.edge_attr is not None:
                    edge_attr = data.edge_attr.to(self.device)
            
            try:
                out = self.model(x, edge_index, edge_attr)
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

        print("[Checkpoint] Loading training data with RMF features and dual edge weights")
        self.data_train = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.TRAIN_EDGE_C,
            edge_path_l=Config.TRAIN_EDGE_L,
            label_path=Config.TRAIN_LABEL,
            rmf_path=Config.TRAIN_RMF,
            churner_set_m1=self.churned_users
        )
        GraphDataProcessor.get_class_distribution(self.data_train)
        
        print("[Checkpoint] Loading validation data with RMF features and dual edge weights")
        self.data_val = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.VAL_EDGE_C,
            edge_path_l=Config.VAL_EDGE_L,
            label_path=Config.VAL_LABEL,
            rmf_path=Config.VAL_RMF,
            churner_set_m1=self.churned_users
        )

        print("[Checkpoint] Loading test data with RMF features and dual edge weights")
        self.data_test = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.TEST_EDGE_C,
            edge_path_l=Config.TEST_EDGE_L,
            label_path=Config.TEST_LABEL,
            rmf_path=Config.TEST_RMF,
            churner_set_m1=self.churned_users
        )
        
        print("[Checkpoint] Experiment initialization complete")

    def create_config_name(self, config):
        """Create a descriptive name for the hyperparameter configuration"""
        # Extract hyperparameters
        lr = config.learning_rate
        hidden = config.hidden_channels
        layers = config.num_layers
        alpha = config.focal_alpha
        gamma = config.focal_gamma
        
        # Format learning rate for readability
        lr_str = f"{lr:.0e}" if lr < 0.01 else f"{lr:.2f}".rstrip('0').rstrip('.')
        
        # Create concatenated name
        config_name = (
            f"lr{lr_str}_"
            f"h{hidden}_"
            f"L{layers}_"
            f"a{alpha}_"
            f"g{gamma}"
        )
        
        return config_name

    def delete_all_previous_runs(self, project_name="gcn-long"):
        """Delete all previous runs in the WandB project before starting new sweep"""
        try:
            print("[Checkpoint] ====== Deleting Previous Runs ======")
            
            # Initialize WandB API
            api = wandb.Api()
            
            # Get all runs from the project
            runs = api.runs(f"{api.default_entity}/{project_name}")
            
            if len(runs) == 0:
                print("[Checkpoint] No previous runs found to delete")
                return
            
            print(f"[Checkpoint] Found {len(runs)} previous runs to delete...")
            
            # Delete each run
            deleted_count = 0
            for run in runs:
                try:
                    run.delete()
                    deleted_count += 1
                    print(f"[Checkpoint] Deleted run: {run.name} ({run.id})")
                except Exception as e:
                    print(f"[Checkpoint] Warning: Could not delete run {run.id}: {str(e)}")
            
            print(f"[Checkpoint] Successfully deleted {deleted_count} runs")
            print("[Checkpoint] ====== Cleanup Complete ======")
            
        except Exception as e:
            print(f"[Checkpoint] Warning: Error during cleanup: {str(e)}")
            print("[Checkpoint] Continuing with sweep despite cleanup error...")

    def train_model(self):
        """Single training run for wandb sweep"""
        # Initialize wandb run
        run = wandb.init()
        config = wandb.config
        
        # Create descriptive configuration name
        config_name = self.create_config_name(config)
        
        # Update run name with configuration
        try:
            if wandb.run:
                wandb.run.name = config_name
        except:
            pass  # Ignore naming errors
        
        # Also log the config name as a metric for easy filtering
        wandb.log({"config_name": config_name}, step=0)
        
        print(f"[Checkpoint] ====== Training Configuration ======")
        print(f"[Checkpoint] Run Name: {config_name}")
        print(f"[Checkpoint] lr={config.learning_rate}, hidden={config.hidden_channels}, layers={config.num_layers}")
        print(f"[Checkpoint] alpha={config.focal_alpha}, gamma={config.focal_gamma}")
        
        try:
            num_features = self.data_train.x.shape[1]
            
            model = EnhancedGCN(
                input_dim=num_features,
                embedding_dim=Config.EMBEDDING_DIM,
                hidden_channels=config.hidden_channels, 
                num_layers=config.num_layers,
                num_nodes=self.data_train.num_nodes,
                dropout_rate=Config.DROPOUT_RATE
            )
            
            trainer = GCNTrainer(model, Config.DEVICE, alpha=config.focal_alpha, gamma=config.focal_gamma)
            
            # Use Adam with weight decay to prevent overfitting and improve stability
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate,
                weight_decay=1e-5  # Add weight decay
            )

            current_best_val_auc = 0
            epochs_no_improve = 0
            nan_epochs = 0  # Count consecutive NaN epochs

            print(f"[Checkpoint] Starting training for up to {Config.MAX_EPOCHS} epochs")
            
            for epoch in range(1, Config.MAX_EPOCHS + 1):
                loss = trainer.train(self.data_train, optimizer)
                
                # Log training loss
                if not np.isnan(loss):
                    wandb.log({"epoch": epoch, "train_loss": loss}, step=epoch)
                    
                if np.isnan(loss):
                    nan_epochs += 1
                    print(f"[Checkpoint] NaN loss detected, nan_epochs={nan_epochs}")
                    if nan_epochs >= 3:
                        print("[Checkpoint] Too many NaN epochs, stopping training")
                        break
                else:
                    nan_epochs = 0  # Reset counter on successful epoch

                if epoch % 5 == 0 and not np.isnan(loss):
                    # Evaluate on validation set
                    val_metrics = trainer.evaluate(self.data_val, calculate_emp=True)
                    
                    # Also evaluate on training set for comparison
                    train_metrics = trainer.evaluate(self.data_train, calculate_emp=False)
                    
                    # Log comprehensive metrics to wandb every 5 epochs
                    log_metrics = {
                        "epoch": epoch,
                        "config_name": config_name,
                        # Training metrics
                        "train_loss_eval": train_metrics['loss'],
                        "train_auc": train_metrics['auc'],
                        "train_auprc": train_metrics['auprc'],
                        "train_lift_0005": train_metrics['lift_0005'],
                        "train_lift_001": train_metrics['lift_001'],
                        "train_lift_005": train_metrics['lift_005'],
                        "train_lift_01": train_metrics['lift_01'],
                        # Validation metrics
                        "val_loss": val_metrics['loss'],
                        "val_auc": val_metrics['auc'],
                        "val_auprc": val_metrics['auprc'],
                        "val_lift_0005": val_metrics['lift_0005'],
                        "val_lift_001": val_metrics['lift_001'],
                        "val_lift_005": val_metrics['lift_005'],
                        "val_lift_01": val_metrics['lift_01'],
                        "val_emp": val_metrics['emp'],
                        "val_mp": val_metrics['mp'],
                        # Gap metrics for overfitting detection
                        "auc_gap": train_metrics['auc'] - val_metrics['auc'],
                        "auprc_gap": train_metrics['auprc'] - val_metrics['auprc']
                    }
                    wandb.log(log_metrics, step=epoch)
                    
                    # Early stopping based on validation AUC
                    if val_metrics['auc'] > current_best_val_auc:
                        current_best_val_auc = val_metrics['auc']
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= Config.PATIENCE // 5:  # Check every 5 epochs
                            print(f"[Checkpoint] Early stopping at epoch {epoch}")
                            break
            
            # Final test evaluation if training was successful
            if not np.isnan(loss):
                test_metrics = trainer.evaluate(self.data_test, calculate_emp=True)
                
                # Log final test metrics
                final_metrics = {
                    "config_name": config_name,
                    # Final test results
                    "final_test_auc": test_metrics['auc'],
                    "final_test_auprc": test_metrics['auprc'],
                    "final_test_lift_0005": test_metrics['lift_0005'],
                    "final_test_lift_001": test_metrics['lift_001'],
                    "final_test_lift_005": test_metrics['lift_005'],
                    "final_test_lift_01": test_metrics['lift_01'],
                    "final_test_emp": test_metrics['emp'],
                    "final_test_mp": test_metrics['mp'],
                    "training_completed": True
                }
                wandb.log(final_metrics)
                # Safely add tags
                try:
                    if wandb.run and hasattr(wandb.run, 'tags'):
                        wandb.run.tags = ["completed", "success"]
                except:
                    pass  # Ignore tagging errors
                
                print(f"[Checkpoint] Final Test Results:")
                print(f"[Checkpoint] Test AUC: {test_metrics['auc']:.6f}")
                print(f"[Checkpoint] Test AUPRC: {test_metrics['auprc']:.6f}")
                print(f"[Checkpoint] Test EMP: {test_metrics['emp']:.6f}")
            
        except Exception as e:
            print(f"[Checkpoint] Error during training: {str(e)}")
            # Use tags and boolean flags instead of status strings
            if wandb.run and hasattr(wandb.run, 'tags'):
                current_tags = list(wandb.run.tags) if wandb.run.tags else []
                wandb.run.tags = current_tags + ["error", "crashed"]
            wandb.log({
                "training_error": True,
                "error_message_length": len(str(e)),  # Log string length instead of string
                "has_error": True
            })
            
        finally:
            wandb.finish()

    def run_hyperparameter_tuning(self):
        print("[Checkpoint] ====== Starting WandB Hyperparameter Tuning ======")
        
        # Delete all previous runs first
        self.delete_all_previous_runs("gcn-long")
        
        # Initialize WandB sweep
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="gcn-long-lc")
        print(f"[Checkpoint] Created WandB sweep: {sweep_id}")
        
        # Run the sweep agent
        wandb.agent(sweep_id, function=self.train_model, count=None)
        
        return None  # Best model tracking handled by wandb

if __name__ == "__main__":
    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("[Checkpoint] ====== Script Started ======")
    try:
        # Initialize WandB project
        wandb.login()  # Uses your global API key
        
        experiment = Experiment()
        experiment.run_hyperparameter_tuning()
        print("[Checkpoint] ====== Script Finished ======")
    except Exception as e:
        print(f"[Checkpoint] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
