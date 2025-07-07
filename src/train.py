
from auxiliary import loss_function_clamped_old, create_masks, hierarchical_triplet_loss, training_loss, neighborhood_contrastive_loss, enhanced_loss_function, evaluate_embeddings, extract_same_cpc_relationships, extract_parent_child_relationships,load_hyperbolic_inputs, mean_average_precision
from models import VGAE,  EnhancedVGAE, HMI, EarlyStopping, HyperbolicEmbeddingModel, FigureOnlyHyperbolicModel, NPairBatchSampler, ImagePairDataset, collate_npairs, collate_enhanced_batch
from process_graph import process_patent_graph, load_patent_graph

import argparse
import json
import re
import os
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
import geoopt as gt
import random
import time
import geoopt.manifolds.stereographic.math as pmath
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.io as tvio
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score




import torch as th
import geoopt as gt
from geoopt.optim.radam import RiemannianAdam

from scipy.io import arff

from matplotlib.patches import Circle
import matplotlib.cm as cm
import seaborn as sns

import warnings; warnings.simplefilter('ignore')
from torch.utils.data import DataLoader

# Set the random seed
torch.manual_seed(42)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Constants
MIN_NORM = 1e-15
DROPOUT_RATE = 0.1 

def load_model(path, latent_dim, hidden_dim):
    """Load Trained model.

    Load the model from the models folder


    Args:
        path (str)

    Returns:
        ```model``
    """
    # select the initilization parameters base on the mode name
    input_dim = 512
    #pattern = r"d(\d+)_(\d+\.?\d*)"
    #match2 = re.search(pattern2, path)
    #model_name = match2.group(1)
    
    model_class=HMI
    # Initialize the model
    num_figures = 32115
    num_patents = 16059
    num_medium_cpc = 595
    num_big_cpc = 126
    num_main_cpc = 9
    num_labels=num_figures + num_patents + num_medium_cpc + num_big_cpc + num_main_cpc
    model = model_class(input_dim, hidden_dim, latent_dim,num_labels )

    # Load the model's parameters
    model.load_state_dict(torch.load(path))

    # Set the model in evaluation mode
    model.eval()
    print(f"📂 Loaded model from {path}")

    return model


def save_model(model, model_name, hidden_dim, latent_dim, learning_rate,epochs):
    """Save trained model.

    Save the trained model into the models folder


    Args:
        model: The trained PyTorch model to be saved
        model_name (str): Name identifier for the model
        latent_dim (int): Dimension of the latent space used in the model
        learning_rate (float): Learning rate used during training
    """
    # Save the model with the corresponding name
    torch.save(
        model.state_dict(), f"models/{model_name}_{hidden_dim}_d{latent_dim}_l{learning_rate}_{epochs}"
    )
    print(f"💾 Model {model_name}_{hidden_dim}_d{latent_dim}_l{learning_rate}_{epochs} saved")

def load_hierarchical_pairs(filepath='../src/hierarchical_pairs.pkl'):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            hierarchical_pairs = pickle.load(f)
        print(f"Loaded {len(hierarchical_pairs)} hierarchical pairs from {filepath}")
        return hierarchical_pairs
    else:
        print(f"File {filepath} not found. Need to extract hierarchical pairs first.")
        return None



def train_pair_classification_model(model, X, A_tilde, pairs_data, hidden_dim, latent_dim, 
                                   epochs=100, lr=0.0001, batch_size=128):
    """
    Train an EnhancedVGAE model to classify pairs of figures based on their connection level.
    
    Args:
        model: The EnhancedVGAE model
        X: Node features
        A_tilde: Adjacency matrix (used for message passing)
        pairs_data: List of tuples (fig1_idx, fig2_idx, level) where level is 1-5
        hidden_dim: Hidden dimension size
        latent_dim: Latent dimension size
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for pair training
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to the appropriate device
    
    model = model.to(device).float()
    
    

    X = X.to(device).float()
    A_tilde = A_tilde.to(device).float()
    # Load the figure_to_row mapping from the pickle file
    with open('../notebooks/image_index_2018.pkl', 'rb') as f:
        figure_to_row = pickle.load(f)
    # Create a mapping from figure names to indices
    
    figure_to_idx = {fig_name: idx for idx, fig_name in enumerate(figure_to_row.keys())}
    
    # Convert pairs data to indices and labels
    pair_indices = []
    pair_labels = []
    
    for fig1, fig2, level in pairs_data:
        if fig1 in figure_to_idx and fig2 in figure_to_idx:
            pair_indices.append((figure_to_idx[fig1], figure_to_idx[fig2]))
            pair_labels.append(level - 1)  # Convert 1-5 to 0-4 for class indices
    
    # Convert to tensors
    pair_indices = torch.tensor(pair_indices, dtype=torch.long)
    pair_labels = torch.tensor(pair_labels, dtype=torch.long)
    
    # Split into train/val/test
    num_pairs = len(pair_indices)
    print(num_pairs)
    indices = torch.randperm(num_pairs)
    
    train_size = int(0.8 * num_pairs)
    val_size = int(0.1 * num_pairs)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        pair_indices[train_indices], pair_labels[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        pair_indices[val_indices], pair_labels[val_indices]
    )
    test_dataset = torch.utils.data.TensorDataset(
        pair_indices[test_indices], pair_labels[test_indices]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
 # Much higher weight for Class 1
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    
    print(f"🚴‍♂️ Training {model.__class__.__name__} on {device}...")
    print(f"Number of training pairs: {len(train_dataset)}")
    print(f"Number of validation pairs: {len(val_dataset)}")
    print(f"Number of test pairs: {len(test_dataset)}")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # For early stopping
    
  
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_pairs, batch_labels in train_loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            # Get embeddings for the pairs
            idx1 = batch_pairs[:, 0]
            idx2 = batch_pairs[:, 1]

            all_embeddings = model(X, A_tilde)
            #all_embeddings = X

            z1 = all_embeddings[idx1]
            z2 = all_embeddings[idx2]
            
            
            logits = model.classify_pair(z1, z2)
            # Forward pass through classifier
            
            logits = logits.float()      
            # Compute loss
            batch_labels = batch_labels.long() 
            loss = loss_fn(logits, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * batch_labels.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_pairs, batch_labels in val_loader:
                batch_pairs = batch_pairs.to(device)
                batch_labels = batch_labels.to(device)
                
                # Get embeddings for the pairs
                idx1 = batch_pairs[:, 0]
                idx2 = batch_pairs[:, 1]
                
                z1 = all_embeddings[idx1]
                z2 = all_embeddings[idx2]
                
                # Forward pass through classifier
                logits = model.classify_pair(z1, z2)
                
                # Compute loss
                loss = loss_fn(logits, batch_labels)
                
                # Track metrics
                val_loss += loss.item() * batch_labels.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        # Calculate average metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_pair_classifier.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_pair_classifier.pt'))
    
    # Test phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Confusion matrix
    confusion = torch.zeros(5, 5, dtype=torch.long)
    
    with torch.no_grad():
        for batch_pairs, batch_labels in test_loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            # Get embeddings for the pairs
            idx1 = batch_pairs[:, 0]
            idx2 = batch_pairs[:, 1]
            
            z1 = all_embeddings[idx1]
            z2 = all_embeddings[idx2]
            
            # Forward pass through classifier
            logits = model.classify_pair(z1, z2)
            
            # Compute loss
            loss = loss_fn(logits, batch_labels)
            
            # Track metrics
            test_loss += loss.item() * batch_labels.size(0)
            pred = logits.argmax(dim=1)
            test_correct += (pred == batch_labels).sum().item()
            test_total += batch_labels.size(0)
            
            # Update confusion matrix
            for t, p in zip(batch_labels.cpu(), pred.cpu()):
                confusion[t, p] += 1
    
    # Calculate test metrics
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Calculate per-class metrics
    for i in range(5):
        precision = confusion[i, i] / confusion[:, i].sum() if confusion[:, i].sum() > 0 else 0
        recall = confusion[i, i] / confusion[i, :].sum() if confusion[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Class {i+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return model



def sample_to_prototype_loss(fig_emb, pos_label_emb, neg_label_emb, k):
    """
    Calculates a sample-to-prototype loss based on hyperbolic distance.
    Uses a formulation equivalent to Nickel & Kiela style log-likelihood loss,
    implemented via cross_entropy for better numerical stability.

    Args:
        fig_emb: Encoded figure embeddings (batch_size, embed_dim).
        pos_label_emb: Embeddings of positive labels (batch_size, embed_dim).
        neg_label_emb: Embeddings of negative labels (batch_size * num_neg, embed_dim).
        k: Curvature tensor.

    Returns:
        Loss tensor.
    """
    batch_size = fig_emb.shape[0]
    num_neg = neg_label_emb.shape[0] // batch_size

    dist_pos = pmath.dist(fig_emb, pos_label_emb, k=k)  # Shape: (batch_size,)

    fig_emb_rep = fig_emb.repeat_interleave(num_neg, dim=0)  # Shape: (batch_size * num_neg, embed_dim)
    dist_neg = pmath.dist(fig_emb_rep, neg_label_emb, k=k)  # Shape: (batch_size * num_neg,)
    dist_neg = dist_neg.view(batch_size, num_neg)  # Shape: (batch_size, num_neg)

    logits_pos = -dist_pos.unsqueeze(1)  # Shape: (batch_size, 1)
    logits_neg = -dist_neg  # Shape: (batch_size, num_neg)

    all_logits = torch.cat([logits_pos, logits_neg], dim=1)  # Shape: (batch_size, 1 + num_neg)

    targets = torch.zeros(batch_size, dtype=torch.long, device=fig_emb.device)

    loss = F.cross_entropy(all_logits, targets)

    return loss
def train_end_to_end_with_hierarchical_model(
    clip_model,
    hyperbolic_model,
    train_dataloader,
    val_dataloader,
    implication_tensor,  # Label hierarchy tensor
    epochs=10,
    clip_lr=1e-5,
    hyperbolic_lr=1e-3,
    temperature=0.07,
    device=None,
    save_dir="models",
    patience=5,
    clip_finetune=True,      # Whether to fine-tune CLIP or keep it frozen
    clip_weight=1.0,         # Weight for CLIP loss
    hyperbolic_weight=1.0,   # Weight for hyperbolic contrastive loss
    retrieval_weight=0.5,    # Weight for sample-to-prototype loss
    hierarchical_weight=0.3, # Weight for hierarchical insideness loss
    figure_pair_weight=0.5,  # Weight for figure-to-figure loss
    reg_weight=0.1           # Weight for regularization loss
):
    """
    End-to-end training of CLIP and hyperbolic embedding models with the comprehensive HyperbolicEmbeddingModel
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models and tensors to device
    clip_model = clip_model.to(device)
    hyperbolic_model = hyperbolic_model.to(device)
    hyperbolic_model.k = hyperbolic_model.k.to(device)
    
    # Convert implication tensor to pairs for the model
    implication_pairs = torch.nonzero(implication_tensor, as_tuple=False)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up optimizers
    if clip_finetune:
        clip_optimizer = torch.optim.AdamW(clip_model.parameters(), lr=clip_lr)
    hyperbolic_optimizer = torch.optim.Adam(hyperbolic_model.parameters(), lr=hyperbolic_lr)
    
    # Set up early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # --- Training ---
        clip_model.train() if clip_finetune else clip_model.eval()
        hyperbolic_model.train()
        
        total_loss = 0.0
        clip_loss_sum = 0.0
        hyperbolic_contrastive_loss_sum = 0.0
        retrieval_loss_sum = 0.0
        hierarchical_loss_sum = 0.0
        figure_pair_loss_sum = 0.0
        reg_loss_sum = 0.0
        batch_count = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} Training"):
            if batch is None:
                continue
            
            # Unpack batch data
            images = batch['images'].to(device)
            n = batch['n']  # Number of pairs
            
            # Get patent labels if available
            pos_patents = batch.get('pos_patents', None)
            neg_patents = batch.get('neg_patents', None)
            
            # Get figure pairs if available
            pos_fig_pairs = batch.get('pos_fig_pairs', None)
            neg_fig_pairs = batch.get('neg_fig_pairs', None)
            
            # Split into anchors and positives for contrastive learning
            anchor_images = images[:n]
            positive_images = images[n:]
            
            # --- Forward pass through CLIP ---
            with torch.set_grad_enabled(clip_finetune):
                # Get CLIP image features
                clip_features = clip_model.get_image_features(pixel_values=images)
                
                # Normalize features
                clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                
                # Split features
                anchor_clip_features = clip_features[:n]
                positive_clip_features = clip_features[n:]
                
                # Compute CLIP contrastive loss (cosine similarity based)
                logits_per_image = torch.matmul(anchor_clip_features, positive_clip_features.t()) / temperature
                labels = torch.arange(n, device=device)
                clip_loss = (F.cross_entropy(logits_per_image, labels) + 
                             F.cross_entropy(logits_per_image.t(), labels)) / 2
            
            # --- Forward pass through hyperbolic model ---
            # Use the comprehensive forward method of the hyperbolic model
            hyperbolic_embeddings, pair_loss, inside_loss, disjoint_loss, label_reg, instance_reg = hyperbolic_model(
                clip_features, 
                implication_pairs=implication_pairs,
                exclusion_pairs=None,  # Add if you have exclusion pairs
                positive_pairs=pos_fig_pairs,
                negative_pairs=neg_fig_pairs
            )
            
            # Split hyperbolic embeddings for contrastive loss
            anchor_hyperbolic = hyperbolic_embeddings[:n]
            positive_hyperbolic = hyperbolic_embeddings[n:]

            # Compute hyperbolic contrastive loss
            hyp_contrastive_loss = hyperbolic_contrastive_loss(
                anchor_hyperbolic, positive_hyperbolic, 
                k=hyperbolic_model.k, temperature=temperature
            )
            
            # Compute retrieval loss if patent labels are available
            retrieval_loss = torch.tensor(0.0, device=device)
            if pos_patents is not None and neg_patents is not None:
                # Get corresponding label embeddings
                pos_label_emb = hyperbolic_model.label_emb[pos_patents]
                neg_label_emb = hyperbolic_model.label_emb[neg_patents]
                
                retrieval_loss = sample_to_prototype_loss(
                    hyperbolic_embeddings, pos_label_emb, neg_label_emb, hyperbolic_model.k
                )
            
            # Combine all losses with their respective weights
            reg_loss = label_reg + instance_reg
            hierarchical_loss = inside_loss + disjoint_loss
            
            loss = (
                clip_weight * clip_loss + 
                hyperbolic_weight * hyp_contrastive_loss +
                retrieval_weight * retrieval_loss +
                hierarchical_weight * hierarchical_loss +
                figure_pair_weight * pair_loss +
                reg_weight * reg_loss
            )
            
            # Backward and optimize
            if clip_finetune:
                clip_optimizer.zero_grad()
            hyperbolic_optimizer.zero_grad()
            
            loss.backward()
            
            if clip_finetune:
                clip_optimizer.step()
            hyperbolic_optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            clip_loss_sum += clip_loss.item()
            hyperbolic_contrastive_loss_sum += hyp_contrastive_loss.item()
            retrieval_loss_sum += retrieval_loss.item()
            hierarchical_loss_sum += hierarchical_loss.item()
            figure_pair_loss_sum += pair_loss.item()
            reg_loss_sum += reg_loss.item()
            batch_count += 1
            
            # Print progress
            if batch_count % 10 == 0:
                print(f"Batch {batch_count}, Total Loss: {loss.item():.4f}")
                print(f"  CLIP Loss: {clip_loss.item():.4f}")
                print(f"  Hyperbolic Contrastive Loss: {hyp_contrastive_loss.item():.4f}")
                print(f"  Retrieval Loss: {retrieval_loss.item():.4f}")
                print(f"  Hierarchical Loss: {hierarchical_loss.item():.4f}")
                print(f"  Figure Pair Loss: {pair_loss.item():.4f}")
                print(f"  Regularization Loss: {reg_loss.item():.4f}")
        
        # Calculate average losses
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_clip_loss = clip_loss_sum / batch_count if batch_count > 0 else 0
        avg_hyp_contrastive_loss = hyperbolic_contrastive_loss_sum / batch_count if batch_count > 0 else 0
        avg_retrieval_loss = retrieval_loss_sum / batch_count if batch_count > 0 else 0
        avg_hierarchical_loss = hierarchical_loss_sum / batch_count if batch_count > 0 else 0
        avg_figure_pair_loss = figure_pair_loss_sum / batch_count if batch_count > 0 else 0
        avg_reg_loss = reg_loss_sum / batch_count if batch_count > 0 else 0
        
        print(f"Epoch {epoch} Training - Avg Loss: {avg_loss:.4f}")
        print(f"  Avg CLIP Loss: {avg_clip_loss:.4f}")
        print(f"  Avg Hyperbolic Contrastive Loss: {avg_hyp_contrastive_loss:.4f}")
        print(f"  Avg Retrieval Loss: {avg_retrieval_loss:.4f}")
        print(f"  Avg Hierarchical Loss: {avg_hierarchical_loss:.4f}")
        print(f"  Avg Figure Pair Loss: {avg_figure_pair_loss:.4f}")
        print(f"  Avg Regularization Loss: {avg_reg_loss:.4f}")
        
        # --- Validation ---
        clip_model.eval()
        hyperbolic_model.eval()

        val_total_loss = 0.0
        val_clip_loss_sum = 0.0
        val_hyp_contrastive_loss_sum = 0.0
        val_retrieval_loss_sum = 0.0
        val_hierarchical_loss_sum = 0.0
        val_figure_pair_loss_sum = 0.0
        val_reg_loss_sum = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} Validation"):
                if batch is None:
                    continue
                
                # Unpack batch data
                images = batch['images'].to(device)
                n = batch['n']  # Number of pairs
                
                # Get patent labels if available
                pos_patents = batch.get('pos_patents', None)
                neg_patents = batch.get('neg_patents', None)
                
                # Get figure pairs if available
                pos_fig_pairs = batch.get('pos_fig_pairs', None)
                neg_fig_pairs = batch.get('neg_fig_pairs', None)
                
                # Forward pass through CLIP
                clip_features = clip_model.get_image_features(pixel_values=images)
                clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                
                anchor_clip_features = clip_features[:n]
                positive_clip_features = clip_features[n:]
                
                # CLIP contrastive loss
                logits_per_image = torch.matmul(anchor_clip_features, positive_clip_features.t()) / temperature
                labels = torch.arange(n, device=device)
                val_clip_loss = (F.cross_entropy(logits_per_image, labels) + 
                                F.cross_entropy(logits_per_image.t(), labels)) / 2
                
                # Forward pass through hyperbolic model
                val_hyperbolic_embeddings, val_pair_loss, val_inside_loss, val_disjoint_loss, val_label_reg, val_instance_reg = hyperbolic_model(
                    clip_features, 
                    implication_pairs=implication_pairs,
                    exclusion_pairs=None,
                    positive_pairs=pos_fig_pairs,
                    negative_pairs=neg_fig_pairs
                )
                
                # Split hyperbolic embeddings for contrastive loss
                val_anchor_hyperbolic = val_hyperbolic_embeddings[:n]
                val_positive_hyperbolic = val_hyperbolic_embeddings[n:]
                
                # Compute hyperbolic contrastive loss
                val_hyp_contrastive_loss = hyperbolic_contrastive_loss(
                    val_anchor_hyperbolic, val_positive_hyperbolic, 
                    k=hyperbolic_model.k, temperature=temperature
                )
                
                # Compute retrieval loss if patent labels are available
                val_retrieval_loss = torch.tensor(0.0, device=device)
                if pos_patents is not None and neg_patents is not None:
                    # Get corresponding label embeddings
                    val_pos_label_emb = hyperbolic_model.label_emb[pos_patents]
                    val_neg_label_emb = hyperbolic_model.label_emb[neg_patents]
                    
                    val_retrieval_loss = sample_to_prototype_loss(
                        val_hyperbolic_embeddings, val_pos_label_emb, val_neg_label_emb, hyperbolic_model.k
                    )
                
                # Combine all losses
                val_reg_loss = val_label_reg + val_instance_reg
                val_hierarchical_loss = val_inside_loss + val_disjoint_loss
                
                val_loss = (
                    clip_weight * val_clip_loss + 
                    hyperbolic_weight * val_hyp_contrastive_loss +
                    retrieval_weight * val_retrieval_loss +
                    hierarchical_weight * val_hierarchical_loss +
                    figure_pair_weight * val_pair_loss +
                    reg_weight * val_reg_loss
                )
                
                # Track losses
                val_total_loss += val_loss.item()
                val_clip_loss_sum += val_clip_loss.item()
                val_hyp_contrastive_loss_sum += val_hyp_contrastive_loss.item()
                val_retrieval_loss_sum += val_retrieval_loss.item()
                val_hierarchical_loss_sum += val_hierarchical_loss.item()
                val_figure_pair_loss_sum += val_pair_loss.item()
                val_reg_loss_sum += val_reg_loss.item()
                val_batch_count += 1

        # Calculate average validation losses
        avg_val_loss = val_total_loss / val_batch_count if val_batch_count > 0 else 0
        avg_val_clip_loss = val_clip_loss_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_hyp_contrastive_loss = val_hyp_contrastive_loss_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_retrieval_loss = val_retrieval_loss_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_hierarchical_loss = val_hierarchical_loss_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_figure_pair_loss = val_figure_pair_loss_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_reg_loss = val_reg_loss_sum / val_batch_count if val_batch_count > 0 else 0

        print(f"Epoch {epoch} Validation - Avg Loss: {avg_val_loss:.4f}")
        print(f"  Avg CLIP Loss: {avg_val_clip_loss:.4f}")
        print(f"  Avg Hyperbolic Contrastive Loss: {avg_val_hyp_contrastive_loss:.4f}")
        print(f"  Avg Retrieval Loss: {avg_val_retrieval_loss:.4f}")
        print(f"  Avg Hierarchical Loss: {avg_val_hierarchical_loss:.4f}")
        print(f"  Avg Figure Pair Loss: {avg_val_figure_pair_loss:.4f}")
        print(f"  Avg Regularization Loss: {avg_val_reg_loss:.4f}")
        # Save models and check for early stopping
        if avg_val_loss < best_val_loss:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving models...")
            
            # Save CLIP model
            if clip_finetune:
                clip_save_path = os.path.join(save_dir, f"clip_model_epoch_{epoch}.pkl")
                with open(clip_save_path, 'wb') as f:
                    pickle.dump(clip_model.state_dict(), f)
            
            # Save hyperbolic model
            hyperbolic_save_path = os.path.join(save_dir, f"hyperbolic_model_epoch_{epoch}.pt")
            torch.save(hyperbolic_model.state_dict(), hyperbolic_save_path)
            
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch  # Track the best epoch
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best models
    if patience_counter < patience:  # If we didn't trigger early stopping
        best_epoch = epoch
    else:
        best_epoch = epoch - patience_counter
    
    
    return clip_model, hyperbolic_model
def extract_mappings_from_adjacency_matrix(
    adjacency_matrix_path,
    figure_paths,
    label_mapping_path=None,
    figure_pattern=r'([A-Z0-9]+)-(\d+)-([A-Z0-9]+)_(\d+)',
    output_dir="data/mappings"
):
    """
    Extract figure-to-patent and patent-to-label mappings from an adjacency matrix.
    
    Args:
        adjacency_matrix_path: Path to the adjacency matrix file (CSV or numpy format)
        figure_paths: List of paths to figure images
        label_mapping_path: Optional path to a file mapping indices to label names
        figure_pattern: Regex pattern to extract patent ID from figure filename
        output_dir: Directory to save the extracted mappings
        
    Returns:
        figure_to_patent: Dictionary mapping figure names to patent IDs
        patent_to_label: Dictionary mapping patent IDs to label indices
        label_to_idx: Dictionary mapping label names to indices
        idx_to_label: Dictionary mapping indices to label names
    """
    import os
    import re
    import json
    import numpy as np
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load adjacency matrix
    if adjacency_matrix_path.endswith('.csv'):
        adj_matrix = np.loadtxt(adjacency_matrix_path, delimiter=',')
    else:
        adj_matrix = np.load(adjacency_matrix_path)
    
    # Create label mappings
    num_labels = adj_matrix.shape[0]
    
    if label_mapping_path:
        # Load label names if provided
        label_mapping = {}
        with open(label_mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    label = parts[1]
                    label_mapping[idx] = label
        
        idx_to_label = label_mapping
        label_to_idx = {v: k for k, v in label_mapping.items()}
    else:
        # Use indices as labels if no mapping provided
        idx_to_label = {i: str(i) for i in range(num_labels)}
        label_to_idx = {str(i): i for i in range(num_labels)}
    
    # Extract patent IDs from figure filenames
    figure_to_patent = {}
    patent_set = set()
    
    for path in figure_paths:
        figure_name = os.path.basename(path)
        match = re.search(figure_pattern, figure_name)
        
        if match:
            # Extract patent ID from filename
            patent_id = match.group(1)  # Assuming first group is the patent ID
            figure_to_patent[figure_name] = patent_id
            patent_set.add(patent_id)
        else:
            print(f"Warning: Could not extract patent ID from figure name: {figure_name}")
    
    print(f"Extracted {len(figure_to_patent)} figure-to-patent mappings")
    print(f"Found {len(patent_set)} unique patents")

    patent_to_label = {}
    for patent in patent_set:
        # Simple hash function to assign patents to labels
        label_idx = sum(ord(c) for c in patent) % num_labels
        patent_to_label[patent] = label_idx
    

    
    print(f"Created {len(patent_to_label)} patent-to-label mappings")
    
    # Convert patent_to_label to use label names instead of indices
    patent_to_label_names = {patent: idx_to_label[label_idx] for patent, label_idx in patent_to_label.items()}
    
    # Save mappings to files
    with open(os.path.join(output_dir, 'figure_to_patent.json'), 'w') as f:
        json.dump(figure_to_patent, f, indent=2)
    
    with open(os.path.join(output_dir, 'patent_to_label_idx.json'), 'w') as f:
        json.dump(patent_to_label, f, indent=2)
    
    with open(os.path.join(output_dir, 'patent_to_label_names.json'), 'w') as f:
        json.dump(patent_to_label_names, f, indent=2)
    
    with open(os.path.join(output_dir, 'label_to_idx.json'), 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    with open(os.path.join(output_dir, 'idx_to_label.json'), 'w') as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)
    
    return figure_to_patent, patent_to_label, label_to_idx, idx_to_label

def build_complete_data_pipeline(
    adjacency_matrix_path,
    image_folder,
    label_mapping_path=None,
    figure_pattern=r'([A-Z0-9]+)-(\d+)-([A-Z0-9]+)_(\d+)',
    output_dir="data/mappings",
    similarity_threshold=0.8  # Threshold for creating positive figure pairs
):
    """
    Build a complete data pipeline from an adjacency matrix.
    
    Args:
        adjacency_matrix_path: Path to the adjacency matrix file
        image_folder: Path to folder containing all images
        label_mapping_path: Optional path to label mapping file
        figure_pattern: Regex pattern to extract patent ID from figure filename
        output_dir: Directory to save the extracted mappings
        similarity_threshold: Threshold for creating positive figure pairs
        
    Returns:
        All necessary data for training the hyperbolic model
    """
    import os
    import json
    import numpy as np
    from pathlib import Path
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    
    # Get all image paths
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths = [
        str(path) for path in Path(image_folder).rglob("*")
        if path.suffix in valid_extensions
    ]
    
    print(f"Found {len(image_paths)} images")
    
    # Extract figure-to-patent and patent-to-label mappings
    figure_to_patent, patent_to_label, label_to_idx, idx_to_label = extract_mappings_from_adjacency_matrix(
        adjacency_matrix_path=adjacency_matrix_path,
        figure_paths=image_paths,
        label_mapping_path=label_mapping_path,
        figure_pattern=figure_pattern,
        output_dir=output_dir
    )
    
    # Create figure-to-positive-figures mapping
    # Method 1: Figures from the same patent are positive pairs
    figure_to_pos_figures = {}
    patent_to_figures = {}
    
    # Group figures by patent
    for figure_name, patent_id in figure_to_patent.items():
        if patent_id not in patent_to_figures:
            patent_to_figures[patent_id] = []
        patent_to_figures[patent_id].append(figure_name)
    
    # Create positive pairs for each figure
    for patent_id, figures in patent_to_figures.items():
        if len(figures) >= 2:
            for i, figure in enumerate(figures):
                if figure not in figure_to_pos_figures:
                    figure_to_pos_figures[figure] = []
                # Add all other figures from the same patent as positives
                figure_to_pos_figures[figure].extend([f for f in figures if f != figure])
    

    use_clip_for_similarity = False
    if use_clip_for_similarity:
        # Load CLIP model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create a mapping from figure name to path
        figure_name_to_path = {os.path.basename(path): path for path in image_paths}
        
        # Process images in batches
        batch_size = 32
        all_figure_names = list(figure_name_to_path.keys())
        all_embeddings = []
        
        for i in range(0, len(all_figure_names), batch_size):
            batch_names = all_figure_names[i:i+batch_size]
            batch_images = [Image.open(figure_name_to_path[name]).convert("RGB") for name in batch_names]
            
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(all_embeddings)
        
        # Create positive pairs based on similarity
        for i, figure_name in enumerate(all_figure_names):
            if figure_name not in figure_to_pos_figures:
                figure_to_pos_figures[figure_name] = []
            
            # Find similar figures
            similar_indices = np.where(similarities[i] > similarity_threshold)[0]
            for idx in similar_indices:
                if idx != i:  # Skip self
                    similar_figure = all_figure_names[idx]
                    # Only add if from the same patent or highly similar
                    if (figure_to_patent.get(figure_name) == figure_to_patent.get(similar_figure) or 
                        similarities[i, idx] > 0.9):
                        figure_to_pos_figures[figure_name].append(similar_figure)
    
    print(f"Created positive pairs for {len(figure_to_pos_figures)} figures")
    
    # Save figure-to-positive-figures mapping
    with open(os.path.join(output_dir, 'figure_to_pos_figures.json'), 'w') as f:
        json.dump(figure_to_pos_figures, f, indent=2)
    
    # Create implication pairs from adjacency matrix
    implication_pairs = []
    for i in range(len(idx_to_label)):
        for j in range(len(idx_to_label)):
            if i != j and adj_matrix[i, j] > 0:
                implication_pairs.append((i, j))
    
    implication_pairs = torch.tensor(implication_pairs, dtype=torch.long)
    
    # Save implication pairs
    np.save(os.path.join(output_dir, 'implication_pairs.npy'), implication_pairs.numpy())
    
    print(f"Created {len(implication_pairs)} implication pairs")
    
    return {
        'image_paths': image_paths,
        'figure_to_patent': figure_to_patent,
        'patent_to_label': patent_to_label,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'figure_to_pos_figures': figure_to_pos_figures,
        'implication_pairs': implication_pairs,
        'num_labels': len(idx_to_label)
    }

def sample_to_prototype_loss(samples, pos_prototypes, neg_prototypes, num_neg_samples, k, margin=0.1, temperature=0.07):
    """
    Compute the sample-to-prototype loss in hyperbolic space.
    
    Args:
        samples: Tensor of shape [batch_size, embed_dim] - sample embeddings
        pos_prototypes: Tensor of shape [batch_size, embed_dim] - positive prototype embeddings
        neg_prototypes: Tensor of shape [batch_size, embed_dim] - negative prototype embeddings
        k: Curvature tensor
        margin: Margin for the triplet loss
        temperature: Temperature for scaling
    
    Returns:
        Tensor: Loss value
    """
    batch_size = samples.size(0)
    embed_dim = samples.size(1)
     # Infer number of negatives per sample
    
    # Reshape negative prototypes to [batch_size, num_neg_samples, embed_dim]
    neg_prototypes = neg_prototypes.view(batch_size, num_neg_samples, embed_dim)
    
    # Compute distances to positive prototypes
    pos_distances = pmath.dist(samples.unsqueeze(1), pos_prototypes.unsqueeze(0), k=k).squeeze(1)  # Shape: [batch_size]
    
    # Compute distances to negative prototypes
    neg_distances = pmath.dist(samples.unsqueeze(1), neg_prototypes, k=k)  # Shape: [batch_size, num_neg_samples]
    
    neg_distances=neg_distances.mean(dim=1) 
    # Compute triplet loss for each negative
    triplet_loss = torch.relu(pos_distances.unsqueeze(1) - neg_distances + margin)  # Shape: [batch_size, num_neg_samples]
    
    # Mean over negatives and batch
    loss = triplet_loss.mean()
    
    return loss

def train_hyperbolic_retrieval_model(
    model,
    X_figures,  # Features only for figures
    Y_pos,  # List of (figure_idx, patent_idx)
    Y_neg,  # List of (figure_idx, patent_idx) - assumes multiple neg per pos
    implication,  # List of (child_label_idx, parent_label_idx)
    exclusion,  # List of (label1_idx, label2_idx)
    label_offsets,  # Dict mapping label type to start index
    positive_figure_pairs,  # New: List of (figure_idx1, figure_idx2) from same patent
    negative_figure_pairs,  # New: List of (figure_idx1, figure_idx2) from different patents
    epochs=60,
    lr=5e-3,
    batch_size=128,
    num_neg_samples=3,  # Number of negative samples per positive pair in a batch
    patience=7,  # Adjusted patience
    constraint_penalty=1e-3,  # Weight for hierarchical loss
    reg_penalty=1e-5,  # Weight for regularization loss
    figure_pair_weight=0.7,
    retrieval_penalty=0.7,  # New: Weight for figure-to-figure cross-entropy loss
    validation_split=0.1,  # Use a fraction of figures for validation
    test_split=0.1,  # Use a fraction of figures for testing
    chunk_size=10000,  # Chunk size for processing constraints
    temperature=0.07,
    use_wandb=True,      # Whether to use Weights & Biases
    wandb_project="hyperbolic-encoder",  # W&B project name
    wandb_run_name=None  # W&B run name (optional)

):
    """
    Train the hyperbolic model for retrieval using sample-to-prototype distance loss
    and cross-entropy loss for figure pairs.
    """
    import numpy as np
    import wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.k = model.k.to(device)  # Ensure curvature tensor is on device
    wandb.login()
    
    # Initialize Weights & Biases
    if use_wandb:
        
        config = {
            "lr": lr,
            "temperature": temperature,
            "constraint_penalty":constraint_penalty,  # Weight for hierarchical loss
            "reg_penalty":reg_penalty,  # Weight for regularization loss
            "figure_pair_weight":figure_pair_weight,
            "retrieval_penalty": retrieval_penalty,
            "patience": patience,
            "epochs":epochs
        }
        wandb.init(entity="alvarodelamaza-vrije-universiteit-amsterdam",project=wandb_project, name=wandb_run_name, config=config)
    # Store temperature in model for cross-entropy loss
    model.temperature = temperature

    num_figures = X_figures.shape[0]
    num_labels = model.label_emb.shape[0]
    print(f"Model label_emb size (LABEL_NUM): {num_labels}")

    patent_start_idx_abs = label_offsets['patents']  # Absolute start index
    patent_end_idx_abs = num_labels + patent_start_idx_abs  # Default if no subsequent offset
    if 'medium_cpcs' in label_offsets:
        patent_end_idx_abs = label_offsets['medium_cpcs']
    # ... (add other elif checks if needed) ...
    num_patents = patent_end_idx_abs - patent_start_idx_abs
    print(f"Derived num_patents: {num_patents} (Absolute index range {patent_start_idx_abs} to {patent_end_idx_abs-1})")
 
    has_figure_pairs = (positive_figure_pairs is not None and len(positive_figure_pairs) > 0) and \
                       (negative_figure_pairs is not None and len(negative_figure_pairs) > 0)
    
    if has_figure_pairs:
        print(f"Processing {len(positive_figure_pairs)} positive and {len(negative_figure_pairs)} negative figure pairs...")
        # Convert to tensors
        pos_fig_pairs_tensor = torch.tensor(positive_figure_pairs, dtype=torch.long)
        neg_fig_pairs_tensor = torch.tensor(negative_figure_pairs, dtype=torch.long)
        
        # Validate figure indices
        pos_fig_pairs_valid = ((0 <= pos_fig_pairs_tensor) & (pos_fig_pairs_tensor < num_figures)).all()
        neg_fig_pairs_valid = ((0 <= neg_fig_pairs_tensor) & (neg_fig_pairs_tensor < num_figures)).all()
        
        if not pos_fig_pairs_valid:
            print("WARNING: Some positive figure pair indices are out of bounds. Filtering...")
            valid_mask = ((0 <= pos_fig_pairs_tensor[:, 0]) & (pos_fig_pairs_tensor[:, 0] < num_figures) & 
                          (0 <= pos_fig_pairs_tensor[:, 1]) & (pos_fig_pairs_tensor[:, 1] < num_figures))
            pos_fig_pairs_tensor = pos_fig_pairs_tensor[valid_mask]
            
        if not neg_fig_pairs_valid:
            print("WARNING: Some negative figure pair indices are out of bounds. Filtering...")
            valid_mask = ((0 <= neg_fig_pairs_tensor[:, 0]) & (neg_fig_pairs_tensor[:, 0] < num_figures) & 
                          (0 <= neg_fig_pairs_tensor[:, 1]) & (neg_fig_pairs_tensor[:, 1] < num_figures))
            neg_fig_pairs_tensor = neg_fig_pairs_tensor[valid_mask]
        
        print(f"After validation: {len(pos_fig_pairs_tensor)} positive and {len(neg_fig_pairs_tensor)} negative figure pairs")
        
        # Group figure pairs by first index for batch processing
        figure_to_pos_figures = defaultdict(list)
        figure_to_neg_figures = defaultdict(list)
        
        for fig1, fig2 in pos_fig_pairs_tensor.tolist():
            figure_to_pos_figures[fig1].append(fig2)
            # Also add the reverse pair for symmetry
            figure_to_pos_figures[fig2].append(fig1)
            
        for fig1, fig2 in neg_fig_pairs_tensor.tolist():
            figure_to_neg_figures[fig1].append(fig2)
            # Also add the reverse pair for symmetry
            figure_to_neg_figures[fig2].append(fig1)
            
        print(f"Grouped figure pairs: {len(figure_to_pos_figures)} figures with positive pairs, "
              f"{len(figure_to_neg_figures)} figures with negative pairs")
    else:
        print("No figure-to-figure pairs provided. Skipping figure pair loss.")
        figure_to_pos_figures = {}
        figure_to_neg_figures = {}

    X_figures_tensor = torch.tensor(X_figures, dtype=torch.float32)

 
    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    exclusion_tensor = torch.tensor(exclusion, dtype=torch.long, device=device) if exclusion else torch.empty((0, 2), dtype=torch.long, device=device)
    # --- Add validation for implication/exclusion relative indices ---
    if implication_tensor.numel() > 0:
        min_imp_idx = implication_tensor.min()
        max_imp_idx = implication_tensor.max()
        print(f"Validation: Implication indices range [{min_imp_idx.item()}, {max_imp_idx.item()}] vs Label Num {num_labels}")
        if min_imp_idx < 0 or max_imp_idx >= num_labels:
             print("ERROR: Invalid relative indices found in implication data!")
             # Handle error
 
    figure_to_pos_patent = {}
    figure_to_neg_patents = {}
    all_figure_indices_with_pairs = set()
    pos_pairs_processed = 0
    pos_pairs_skipped_fig_idx = 0
    pos_pairs_skipped_pat_idx = 0
    neg_pairs_processed = 0
    neg_pairs_skipped_fig_idx = 0
    neg_pairs_skipped_pat_idx = 0

    print("Processing loaded Y_pos...")
    for fig_idx, patent_idx_relative in Y_pos:  # patent_idx_relative is now 0-based
        pos_pairs_processed += 1
        # Check figure index
        if not (0 <= fig_idx < num_figures):
            pos_pairs_skipped_fig_idx += 1
            continue
     
        if not (0 <= patent_idx_relative < num_labels):
            print(f"Warning: Invalid relative patent index {patent_idx_relative} (range 0-{num_labels-1}) found in Y_pos for figure {fig_idx}. Skipping.")
            pos_pairs_skipped_pat_idx += 1
            continue

        patent_start_idx_rel = 0  # Patents are the first labels
        patent_end_idx_rel = num_patents
        if not (patent_start_idx_rel <= patent_idx_relative < patent_end_idx_rel):
             print(f"Note: Y_pos relative patent index {patent_idx_relative} is outside expected patent range [0-{patent_end_idx_rel-1}] but within label range [0-{num_labels-1}]. Accepting.")
             # Decide if this is acceptable or an error
        # --- ---

        figure_to_pos_patent[fig_idx] = patent_idx_relative  # Store relative index
        all_figure_indices_with_pairs.add(fig_idx)

    print(f"Processed {pos_pairs_processed} positive pairs.")
    print(f"  Skipped {pos_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {pos_pairs_skipped_pat_idx} due to invalid relative patent index.")

    print("Processing loaded Y_neg...")
    for fig_idx, patent_idx_relative in Y_neg:  # patent_idx_relative is now 0-based
        neg_pairs_processed += 1
        # Check figure index
        if not (0 <= fig_idx < num_figures):
            neg_pairs_skipped_fig_idx += 1
            continue
        
        if not (0 <= patent_idx_relative < num_labels):
            print(f"Warning: Invalid relative patent index {patent_idx_relative} (range 0-{num_labels-1}) found in Y_neg for figure {fig_idx}. Skipping.")
            neg_pairs_skipped_pat_idx += 1
            continue
        patent_start_idx_rel = 0
        patent_end_idx_rel = num_patents
        if not (patent_start_idx_rel <= patent_idx_relative < patent_end_idx_rel):
             print(f"Note: Y_neg relative patent index {patent_idx_relative} is outside expected patent range [0-{patent_end_idx_rel-1}] but within label range [0-{num_labels-1}]. Accepting.")
  

        if fig_idx not in figure_to_neg_patents:
            figure_to_neg_patents[fig_idx] = []
        figure_to_neg_patents[fig_idx].append(patent_idx_relative)  # Store relative index

    print(f"Processed {neg_pairs_processed} negative pairs.")
    print(f"  Skipped {neg_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {neg_pairs_skipped_pat_idx} due to invalid relative patent index.")

    
    print(f"Populated figure_to_pos_patent with {len(figure_to_pos_patent)} entries.")
    print(f"Populated figure_to_neg_patents with {len(figure_to_neg_patents)} entries.")


    if figure_to_pos_patent:
        first_key = next(iter(figure_to_pos_patent))
        print(f"  Example figure_to_pos_patent entry: ({first_key}, {figure_to_pos_patent[first_key]})")
    if figure_to_neg_patents:
        first_key = next(iter(figure_to_neg_patents))
        num_negs_example = len(figure_to_neg_patents[first_key])
        print(f"  Example figure_to_neg_patents entry: ({first_key}, {figure_to_neg_patents[first_key][:5]}...) Num negs: {num_negs_example}")
        # Check distribution of negative counts
        neg_counts = [len(v) for v in figure_to_neg_patents.values()]
        print(f"  Negative sample counts per figure: Min={min(neg_counts)}, Max={max(neg_counts)}, Avg={np.mean(neg_counts):.2f}, Median={np.median(neg_counts)}")
        print(f"  Number of figures with < {num_neg_samples} negatives: {sum(1 for count in neg_counts if count < num_neg_samples)}")
    
   
    # Convert features to tensor
    X_figures_tensor = torch.tensor(X_figures, dtype=torch.float32)  # Keep on CPU initially

    # Convert constraint lists to tensors on the correct device
    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    exclusion_tensor = torch.tensor(exclusion, dtype=torch.long, device=device) if exclusion else torch.empty((0, 2), dtype=torch.long, device=device)

    # Chunk constraints
    #implication_chunks = [implication_tensor[i:i+chunk_size] for i in range(0, implication_tensor.size(0), chunk_size)]
    exclusion_chunks = [exclusion_tensor[i:i+chunk_size] for i in range(0, exclusion_tensor.size(0), chunk_size)]
    #print(f"Using {len(implication_chunks)} implication chunks and {len(exclusion_chunks)} exclusion chunks.")
    
    trainable_figure_indices = sorted(list(all_figure_indices_with_pairs))
    random.shuffle(trainable_figure_indices)
    # Split indices
    num_trainable = len(trainable_figure_indices)
    val_count = int(num_trainable * validation_split)
    test_count = int(num_trainable * test_split)
    train_count = num_trainable - val_count - test_count

    train_indices = trainable_figure_indices[:train_count]
    val_indices = trainable_figure_indices[train_count : train_count + val_count]
    test_indices = trainable_figure_indices[train_count + val_count :]

    print(f"Total figures with pairs: {num_trainable}")
    print(f"Training figures: {len(train_indices)}, Validation figures: {len(val_indices)}, Test figures: {len(test_indices)}")

    def create_batch_with_figure_pairs(
    indices, batch_size, num_neg, 
    figure_to_pos_patent, figure_to_neg_patents, 
    figure_to_pos_figures, figure_to_neg_figures, 
    X_figures_tensor, device
):
        import random
        random.shuffle(indices)
        batches_yielded = 0
        figures_processed = 0

        for i in range(0, len(indices), batch_size):
            batch_fig_indices = indices[i : i+batch_size]
            figures_processed += len(batch_fig_indices)
            batch_pos_pat_indices = []
            batch_neg_pat_indices = []
            valid_batch_indices = []

            # For figure-to-figure pairs
            batch_pos_fig_pairs = []
            batch_neg_fig_pairs = []

            for fig_idx in batch_fig_indices:
                # Process figure-to-patent pairs
                has_pos = fig_idx in figure_to_pos_patent
                has_negs = fig_idx in figure_to_neg_patents

                if has_pos and has_negs:
                    pos_pat = figure_to_pos_patent[fig_idx]
                    available_negs = figure_to_neg_patents[fig_idx]
                    neg_pats = random.sample(available_negs, min(len(available_negs), num_neg))
                    batch_pos_pat_indices.append(pos_pat)
                    batch_neg_pat_indices.extend(neg_pats)
                    valid_batch_indices.append(fig_idx)

                # Process figure-to-figure pairs
                has_pos_figs = fig_idx in figure_to_pos_figures and figure_to_pos_figures[fig_idx]
                has_neg_figs = fig_idx in figure_to_neg_figures and figure_to_neg_figures[fig_idx]

                if has_pos_figs:
                    pos_fig = random.choice(figure_to_pos_figures[fig_idx])
                    batch_pos_fig_pairs.append((fig_idx, pos_fig))

                if has_neg_figs:
                    neg_fig = random.choice(figure_to_neg_figures[fig_idx])
                    batch_neg_fig_pairs.append((fig_idx, neg_fig))

            if not valid_batch_indices:
                print(f"Skipping batch: No valid indices. Batch figures: {batch_fig_indices}")
                continue  # Skip if batch is empty

            # Handle empty figure pairs
            if not batch_pos_fig_pairs:
                print(f"Warning: No positive figure pairs in batch {batches_yielded}. Adding placeholders.")
                batch_pos_fig_pairs = [(fig_idx, fig_idx) for fig_idx in valid_batch_indices]

            if not batch_neg_fig_pairs:
                print(f"Warning: No negative figure pairs in batch {batches_yielded}. Adding placeholders.")
                batch_neg_fig_pairs = [(fig_idx, fig_idx) for fig_idx in valid_batch_indices]

            # Debugging: Print batch contents
            #print(f"Batch {batches_yielded}: Valid Indices = {valid_batch_indices}, Pos Pairs = {batch_pos_fig_pairs}, Neg Pairs = {batch_neg_fig_pairs}")
            #print(f"X_figures_tensor: {X_figures_tensor}, Type: {type(X_figures_tensor)}")
            #print(f"Valid Batch Indices: {valid_batch_indices}, Type: {type(valid_batch_indices)}, finn")
            # Yield batch
            yield {
                "figures": X_figures_tensor[valid_batch_indices].to(device),
                "pos_patents": torch.tensor(batch_pos_pat_indices, device=device),
                "neg_patents": torch.tensor(batch_neg_pat_indices, device=device),
                "pos_fig_pairs": batch_pos_fig_pairs,
                "neg_fig_pairs": batch_neg_fig_pairs,
            }
            batches_yielded += 1

        
    # --- Optimizer and Early Stopping ---
    optimizer = gt.optim.RiemannianAdam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_figure_pair_val_loss = 1000000  # Track best validation mAP
    optimizer.zero_grad()
    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_retrieval_loss = 0.0
        epoch_hier_loss = 0.0
        epoch_reg_loss=0.0
        epoch_figure_pair_loss = 0.0
        batch_count = 0
        #if epoch==5:
        #    figure_pair_weight=5.0
        #    constraint_penalty=8.0
        #    best_figure_pair_val_loss = 1000000
        #if epoch==10:
        #    reg_penalty=0.1
        # Create batches with figure pairs
        batch_generator = create_batch_with_figure_pairs(
            train_indices, batch_size, num_neg_samples,
            figure_to_pos_patent, figure_to_neg_patents,
            figure_to_pos_figures, figure_to_neg_figures,
            X_figures_tensor, device
        )
        
        for batch in batch_generator:
            batch_x = batch["figures"]
            batch_pos_pat = batch["pos_patents"]
            batch_neg_pat = batch["neg_patents"]
            batch_pos_fig_pairs = batch["pos_fig_pairs"]
            batch_neg_fig_pairs = batch["neg_fig_pairs"] 
            batch_pos_fig_pairs = torch.tensor(batch_pos_fig_pairs, dtype=torch.long, device=device)
            batch_neg_fig_pairs = torch.tensor(batch_neg_fig_pairs, dtype=torch.long, device=device)

# Concatenate tensors
            encoded_figures = model.encode_figures(batch_x)
            # Calculate hierarchical loss (process chunks)
            inside_loss_total = torch.tensor(0.0, device=device)
            disjoint_loss_total = torch.tensor(0.0, device=device)
            
            current_label_emb = model.label_emb
            hierarchical_loss, _ = model.calculate_hierarchical_loss(implication_tensor, None)

            # Calculate regularization loss
            label_reg, instance_reg = model.calculate_reg_loss(encoded_figures)
            reg_loss = label_reg + instance_reg

            # --- Retrieval Loss ---
            # Get corresponding label embeddings
            pos_label_emb = model.label_emb[batch_pos_pat]
            neg_label_emb = model.label_emb[batch_neg_pat]

            retrieval_loss = sample_to_prototype_loss(
                encoded_figures, pos_label_emb, neg_label_emb,num_neg_samples, model.k
            )

            # --- Figure-to-Figure Cross-Entropy Loss ---
            figure_pair_loss = torch.tensor(0.0, device=device)
            if has_figure_pairs and batch_pos_fig_pairs is not None and batch_neg_fig_pairs is not None:
                if batch_pos_fig_pairs.size(0) > 0 and batch_neg_fig_pairs.size(0) > 0:
                    # Combine positive and negative pairs
                    all_pairs = torch.cat([batch_pos_fig_pairs, batch_neg_fig_pairs], dim=0)
                    #print(f"All Pairs Shape: {all_pairs.shape}")
                    
                    # Create labels: 1 for positive pairs, 0 for negative pairs
                    labels = torch.zeros(len(all_pairs), device=device)
                    labels[:batch_pos_fig_pairs.size(0)] = 1.0
                    
                    # Calculate hyperbolic distances between pairs
                    pair_distances = []
                    for pair in all_pairs:
                        idx1, idx2 = pair.tolist()  # Convert tensor to Python integers
                        
                        # Retrieve embeddings directly from X_figures_tensor
                        emb1 = model.encode_figures(X_figures_tensor[idx1].unsqueeze(0).to(device))
                        emb2 = model.encode_figures(X_figures_tensor[idx2].unsqueeze(0).to(device))
                        
                        # Calculate hyperbolic distance
                        dist = pmath.dist(emb1, emb2, k=model.k).squeeze()
                        pair_distances.append(dist)
                    
                    if pair_distances:
                        # Stack distances into a tensor
                        pair_distances = torch.stack(pair_distances)
                        # Convert distances to similarities (smaller distance = higher similarity)
                        similarities = -pair_distances / temperature

                        # Ensure labels tensor is float for BCEWithLogitsLoss
                        figure_pair_loss = F.binary_cross_entropy_with_logits(similarities, labels.float())
                    else:
                        print("no pair distances")
                else:
                    print("no pair distances 2")
            else:
                print(has_figure_pairs, batch_pos_fig_pairs, batch_neg_fig_pairs)
    
             
            total_loss = (
                retrieval_penalty+ retrieval_loss
                + constraint_penalty * hierarchical_loss
                + reg_penalty * reg_loss
                + figure_pair_weight * figure_pair_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_retrieval_loss += retrieval_loss.item()
            epoch_hier_loss += hierarchical_loss.item()
            epoch_reg_loss+= reg_loss.item()
            epoch_figure_pair_loss += figure_pair_loss.item()
            batch_count += 1

            if batch_count % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_count}, Loss: {total_loss.item():.4f}, "
                      f"Retrieval: {retrieval_loss.item():.4f}, Hier: {hierarchical_loss.item():.4f}, "
                      f"Fig Pair: {figure_pair_loss.item():.4f}, Reg: {reg_loss.item():.4f}")
                torch.cuda.empty_cache()
            if use_wandb and batch_count % 10 == 0:
                wandb.log({
                    "batch": batch_count + (epoch-1) * (len(train_indices)/128),
                    "batch_loss": total_loss.item(),
                    "retrieval_loss": retrieval_loss.item(),
                    "hierarchical_loss": hierarchical_loss.item(),
                    "figure_pair_loss": figure_pair_loss.item(),
                    "reg_loss" : reg_loss.item()
                })

        avg_loss = epoch_total_loss / batch_count if batch_count > 0 else 0
        avg_retr_loss = epoch_retrieval_loss / batch_count if batch_count > 0 else 0
        avg_hier_loss = epoch_hier_loss / batch_count if batch_count > 0 else 0
        avg_fig_pair_loss = epoch_figure_pair_loss / batch_count if batch_count > 0 else 0
        avg_reg_loss = epoch_reg_loss/ batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch} Summary - Avg Loss: {avg_loss:.4f} , Avg Retrieval: {avg_retr_loss:.4f}, Avg Hier: {avg_hier_loss:.4f}, Avg Fig Pair: {avg_fig_pair_loss:.4f}, Avg Reg: {avg_reg_loss:.4f}")

        # --- Validation ---
        model.eval()
        # Create batches with figure pairs
        batch_generator_val = create_batch_with_figure_pairs(
            val_indices, batch_size, num_neg_samples,
            figure_to_pos_patent, figure_to_neg_patents,
            figure_to_pos_figures, figure_to_neg_figures,
            X_figures_tensor, device
        )
        epoch_total_val_loss = 0.0
        epoch_retrieval_val_loss = 0.0
        epoch_hier_val_loss = 0.0
        epoch_reg_val_loss=0.0
        epoch_figure_pair_val_loss = 0.0
        batch_val_count=0.0

        
        for batch in batch_generator_val:
            batch_x = batch["figures"]
            batch_pos_pat = batch["pos_patents"]
            batch_neg_pat = batch["neg_patents"]
            batch_pos_fig_pairs = batch["pos_fig_pairs"]
            batch_neg_fig_pairs = batch["neg_fig_pairs"] 
            batch_pos_fig_pairs = torch.tensor(batch_pos_fig_pairs, dtype=torch.long, device=device)
            batch_neg_fig_pairs = torch.tensor(batch_neg_fig_pairs, dtype=torch.long, device=device)
            optimizer.zero_grad()

            # Encode figures
            encoded_figures_val = model.encode_figures(batch_x)

            # Calculate hierarchical loss (process chunks)
            inside_loss_total = torch.tensor(0.0, device=device)
            disjoint_loss_total = torch.tensor(0.0, device=device)
            
            current_label_emb = model.label_emb
            hierarchical_val_loss, _ = model.calculate_hierarchical_loss(implication_tensor, None)

            # Calculate regularization loss
            label_val_reg, instance_val_reg = model.calculate_reg_loss(encoded_figures_val)
            reg_val_loss = label_val_reg + instance_val_reg

   
            pos_label_emb = model.label_emb[batch_pos_pat]
            neg_label_emb = model.label_emb[batch_neg_pat]
            
            retrieval_val_loss = sample_to_prototype_loss(
                        encoded_figures_val, pos_label_emb, neg_label_emb, num_neg_samples, model.k
                    )
            
            

            figure_pair_val_loss = torch.tensor(0.0, device=device)
            if has_figure_pairs and batch_pos_fig_pairs is not None and batch_neg_fig_pairs is not None:
                if batch_pos_fig_pairs.size(0) > 0 and batch_neg_fig_pairs.size(0) > 0:
                    # Combine positive and negative pairs
                    all_pairs = torch.cat([batch_pos_fig_pairs, batch_neg_fig_pairs], dim=0)
                    #print(f"All Pairs Shape: {all_pairs.shape}")
                    
                    # Create labels: 1 for positive pairs, 0 for negative pairs
                    labels = torch.zeros(len(all_pairs), device=device)
                    labels[:batch_pos_fig_pairs.size(0)] = 1.0
                    
                    # Calculate hyperbolic distances between pairs
                    pair_distances_val = []
                    for pair in all_pairs:
                        idx1, idx2 = pair.tolist()  # Convert tensor to Python integers
                        
                        # Retrieve embeddings directly from X_figures_tensor
                        emb1 = model.encode_figures(X_figures_tensor[idx1].unsqueeze(0).to(device))
                        emb2 = model.encode_figures(X_figures_tensor[idx2].unsqueeze(0).to(device))
                        
                        # Calculate hyperbolic distance
                        dist = pmath.dist(emb1, emb2, k=model.k).squeeze()
                        pair_distances_val.append(dist)
                    
                    if pair_distances_val:
                        # Stack distances into a tensor
                        pair_distances_val = torch.stack(pair_distances_val)
                        # Convert distances to similarities (smaller distance = higher similarity)
                        similarities_val = -pair_distances_val / temperature

                        # Ensure labels tensor is float for BCEWithLogitsLoss
                        figure_pair_val_loss = F.binary_cross_entropy_with_logits(similarities_val, labels.float())
                    else:
                        print("no pair distances")
                else:
                    print("no pair distances 2")
            else:
                print(has_figure_pairs, batch_pos_fig_pairs, batch_neg_fig_pairs)

            # --- Total Loss ---
            total_val_loss = (
                retrieval_penalty+ retrieval_val_loss
                + constraint_penalty * hierarchical_val_loss
                + reg_penalty * reg_val_loss
                + figure_pair_weight * figure_pair_val_loss
            )

            epoch_total_val_loss += total_val_loss.item()
            epoch_retrieval_val_loss += retrieval_val_loss.item()
            epoch_hier_val_loss += hierarchical_val_loss.item()
            epoch_figure_pair_val_loss += figure_pair_val_loss.item()
            epoch_reg_val_loss += reg_val_loss.item()
            batch_val_count += 1

        avg_val_loss = epoch_total_val_loss / batch_val_count if batch_val_count > 0 else 0
        avg_val_retr_loss = epoch_retrieval_val_loss / batch_val_count if batch_val_count > 0 else 0
        avg_val_hier_loss = epoch_hier_val_loss / batch_val_count if batch_val_count > 0 else 0
        avg_val_fig_pair_loss = epoch_figure_pair_val_loss / batch_val_count if batch_val_count > 0 else 0
        avg_val_reg_loss = epoch_reg_val_loss / batch_val_count if batch_val_count > 0 else 0
        
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f} , Avg Retrieval: {avg_val_retr_loss:.4f}, Avg Hier: {avg_val_hier_loss:.4f}, Avg Fig Pair: {avg_val_fig_pair_loss:.4f}, Avg Reg: {avg_val_reg_loss:.4f}")

        if use_wandb:
            wandb.log({
                "batch": epoch*  (len(train_indices)/128) ,
                "batch_loss": avg_loss,
                "retrieval_loss": avg_retr_loss,
                "hierarchical_loss": avg_hier_loss,
                "figure_pair_loss" : figure_pair_loss,
                "reg_loss" :avg_reg_loss,
                "val_loss": avg_val_loss,
                "val_retrieval_loss": avg_val_retr_loss,
                "val_hierarchical_loss": avg_val_hier_loss,
                "val_figure_pair_loss": avg_val_fig_pair_loss,
                "val_reg_loss" :reg_val_loss
            })
        
        if  avg_val_loss < best_figure_pair_val_loss:
            best_figure_pair_val_loss = avg_val_loss
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), f'best_retrieval_model_c{model.c}_e{model.embed_dim}.pt')
            early_stopping.counter = 0  # Reset counter if performance improves
        else:
            early_stopping.counter += 1
            print(f'EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}')
            if early_stopping.counter >= early_stopping.patience:
                print("Early stopping triggered.")
                break

        torch.cuda.empty_cache()

    # --- Final Testing ---
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f'best_retrieval_model_c{model.c}_e{model.embed_dim}.pt'))
    model.eval()
    # Create batches with figure pairs
    batch_generator_test = create_batch_with_figure_pairs(
        test_indices, batch_size, num_neg_samples,
        figure_to_pos_patent, figure_to_neg_patents,
        figure_to_pos_figures, figure_to_neg_figures,
        X_figures_tensor, device
    )
    epoch_total_test_loss = 0.0
    epoch_retrieval_test_loss = 0.0
    epoch_hier_test_loss = 0.0
    epoch_figure_pair_test_loss = 0.0
    batch_test_count=0.0

    for batch in batch_generator_test:
        batch_x = batch["figures"]
        batch_pos_pat = batch["pos_patents"]
        batch_neg_pat = batch["neg_patents"]
        batch_pos_fig_pairs = batch["pos_fig_pairs"]
        batch_neg_fig_pairs = batch["neg_fig_pairs"] 
        batch_pos_fig_pairs = torch.tensor(batch_pos_fig_pairs, dtype=torch.long, device=device)
        batch_neg_fig_pairs = torch.tensor(batch_neg_fig_pairs, dtype=torch.long, device=device)
        optimizer.zero_grad()


        # Encode figures
        encoded_figures = model.encode_figures(batch_x)
    
        current_label_emb = model.label_emb
        hierarchical_test_loss, _ = model.calculate_hierarchical_loss(implication_tensor, None)

        # Calculate regularization loss
        label_reg, instance_reg = model.calculate_reg_loss(encoded_figures)
        reg_loss = label_reg + instance_reg

        # --- Retrieval Loss ---
        # Get corresponding label embeddings
        pos_label_emb = model.label_emb[batch_pos_pat]
        neg_label_emb = model.label_emb[batch_neg_pat]

        retrieval_test_loss = sample_to_prototype_loss(
            encoded_figures, pos_label_emb, neg_label_emb, num_neg_samples, model.k
        )

        # --- Figure-to-Figure Cross-Entropy Loss ---
        figure_pair_test_loss = torch.tensor(0.0, device=device)
        
        if has_figure_pairs and batch_pos_fig_pairs is not None and batch_neg_fig_pairs is not None:
            if batch_pos_fig_pairs.size(0) > 0 and batch_neg_fig_pairs.size(0) > 0:
                # Combine positive and negative pairs
                all_pairs = torch.cat([batch_pos_fig_pairs, batch_neg_fig_pairs], dim=0)
                #print(f"All Pairs Shape: {all_pairs.shape}")
                
                # Create labels: 1 for positive pairs, 0 for negative pairs
                labels = torch.zeros(len(all_pairs), device=device)
                labels[:batch_pos_fig_pairs.size(0)] = 1.0
                
                # Calculate hyperbolic distances between pairs
                pair_distances_val = []
                for pair in all_pairs:
                    idx1, idx2 = pair.tolist()  # Convert tensor to Python integers
                    
                    # Retrieve embeddings directly from X_figures_tensor
                    emb1 = model.encode_figures(X_figures_tensor[idx1].unsqueeze(0).to(device))
                    emb2 = model.encode_figures(X_figures_tensor[idx2].unsqueeze(0).to(device))
                    
                    # Calculate hyperbolic distance
                    dist = pmath.dist(emb1, emb2, k=model.k).squeeze()
                    pair_distances_val.append(dist)
                
                if pair_distances_val:
                    # Stack distances into a tensor
                    pair_distances_val = torch.stack(pair_distances_val)
                    # Convert distances to similarities (smaller distance = higher similarity)
                    similarities_val = -pair_distances_val / temperature

                    # Ensure labels tensor is float for BCEWithLogitsLoss
                    figure_pair_test_loss = F.binary_cross_entropy_with_logits(similarities_val, labels.float())
                else:
                    print("no pair distances")
            else:
                print("no pair distances 2")
        else:
            print(has_figure_pairs, batch_pos_fig_pairs, batch_neg_fig_pairs)

        # --- Total Loss ---
        total_test_loss = (
            retrieval_penalty+ retrieval_test_loss
            + constraint_penalty * hierarchical_test_loss
            + reg_penalty * reg_loss
            + figure_pair_weight * figure_pair_test_loss
        )

        epoch_total_test_loss += total_test_loss.item()
        epoch_retrieval_test_loss += retrieval_test_loss.item()
        epoch_hier_test_loss += hierarchical_test_loss.item()
        epoch_figure_pair_test_loss += figure_pair_test_loss.item()
        batch_test_count += 1


    avg_test_loss = epoch_total_test_loss / batch_test_count if batch_test_count > 0 else 0
    avg_test_retr_loss = epoch_retrieval_test_loss / batch_test_count if batch_val_count > 0 else 0
    avg_test_hier_loss = epoch_hier_test_loss / batch_test_count if batch_test_count > 0 else 0
    avg_test_fig_pair_loss = epoch_figure_pair_test_loss / batch_test_count if batch_test_count > 0 else 0
        
    print(f"TestLoss: {avg_test_loss:.4f} , Avg Retrieval: {avg_test_retr_loss:.4f}, Avg Hier: {avg_test_hier_loss:.4f}, Avg Fig Pair: {avg_test_fig_pair_loss:.4f}, Avg Reg: {avg_test_reg_loss:.4f}")

    torch.cuda.empty_cache()
    
        
    wandb.finish()

    return model, avg_test_loss
def create_n_pair_batch(indices, batch_size, figure_to_pos_figures, X_figures_tensor, device):
    """
    For each batch:
        - Sample batch_size anchor figures
        - For each anchor, sample one positive figure (from same patent)
        - The batch is [anchor_1, ..., anchor_B, pos_1, ..., pos_B]
    """
    import random

    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        anchors = indices[i : i + batch_size]
        positives = []
        valid_anchors = []
        for anchor in anchors:
            pos_candidates = figure_to_pos_figures.get(anchor, [])
            if pos_candidates:
                pos = random.choice(pos_candidates)
                positives.append(pos)
                valid_anchors.append(anchor)
        if not valid_anchors:
            continue  # skip empty batch

        # Remove accidental anchor==positive
        batch_pairs = [(a, p) for a, p in zip(valid_anchors, positives) if a != p]
        if not batch_pairs:
            continue

        anchor_indices, pos_indices = zip(*batch_pairs)
        batch_indices = list(anchor_indices) + list(pos_indices)
        batch_x = X_figures_tensor[batch_indices].to(device)
        yield batch_x, len(anchor_indices), list(anchor_indices), list(pos_indices)
import torch
import torch.nn.functional as F
def train_hyperbolic_contrastive(
    model,
    X_figures,
    figure_to_pos_figures,
    train_indices,
    val_indices,
    epochs=1,
    batch_size=128,
    lr=1e-3,
    temperature=0.07,
    device=None,
    save_path="best_model.pt",
    patience=5,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.k = model.k.to(device)
    X_figures_tensor = torch.tensor(X_figures, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        batch_count = 0

        batch_generator = create_n_pair_batch(
            train_indices, batch_size, figure_to_pos_figures, X_figures_tensor, device
        )

        for batch_x, n, anchor_indices, pos_indices in batch_generator:
            encoded = model.encode_figures(batch_x).to(device)
            anchors = encoded[:n].to(device)
            positives = encoded[n:].to(device)

            # Compute pairwise hyperbolic distances: [n, n]
            dists = []
            for i in range(n):
                dists_row = []
                for j in range(n):
                    d = pmath.dist(anchors[i:i+1].to(device), positives[j:j+1].to(device), k=model.k.to(device)).squeeze()
                    dists_row.append(d)
                dists.append(torch.stack(dists_row))
            dist_matrix = torch.stack(dists)  # shape [n, n]

            sim_matrix = -dist_matrix / temperature
            labels = torch.arange(n, device=sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_count}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch} Summary - Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_total_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            val_batch_generator = create_n_pair_batch(
                val_indices, batch_size, figure_to_pos_figures, X_figures_tensor, device
            )
            for batch_x, n, anchor_indices, pos_indices in val_batch_generator:
                encoded = model.encode_figures(batch_x)
                anchors = encoded[:n]
                positives = encoded[n:]

                dists = []
                for i in range(n):
                    dists_row = []
                    for j in range(n):
                        d = pmath.dist(anchors[i:i+1], positives[j:j+1], k=model.k).squeeze()
                        dists_row.append(d)
                    dists.append(torch.stack(dists_row))
                dist_matrix = torch.stack(dists)

                sim_matrix = -dist_matrix / temperature
                labels = torch.arange(n, device=sim_matrix.device)
                loss = F.cross_entropy(sim_matrix, labels)

                val_total_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_total_loss / val_batch_count if val_batch_count > 0 else 0
        print(f"Epoch {epoch} Summary - Avg Val Loss: {avg_val_loss:.4f}")

        # --- Model Saving & Early Stopping ---
        if avg_val_loss < best_val_loss:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded best model from {save_path}")

    return model

def train_hyperbolic_retrieval_model_old(
    model,
    X_figures, # Features only for figures
    Y_pos, # List of (figure_idx, patent_idx)
    Y_neg, # List of (figure_idx, patent_idx) - assumes multiple neg per pos
    implication, # List of (child_label_idx, parent_label_idx)
    exclusion, # List of (label1_idx, label2_idx)
    label_offsets, # Dict mapping label type to start index
    epochs=60,
    lr=5e-3,
    batch_size=128,
    num_neg_samples=3, # Number of negative samples per positive pair in a batch
    patience=7, # Adjusted patience
    constraint_penalty=1e-3, # Weight for hierarchical loss
    reg_penalty=1e-5, # Weight for regularization loss
    validation_split=0.1, # Use a fraction of figures for validation
    test_split=0.1, # Use a fraction of figures for testing
    chunk_size=8000 # Chunk size for processing constraints
):
    """
    Train the hyperbolic model for retrieval using sample-to-prototype distance loss.
    """
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.k = model.k.to(device) # Ensure curvature tensor is on device

        # --- Data Preparation ---
    num_figures = X_figures.shape[0]
    num_labels = model.label_emb.shape[0]
    print(f"Model label_emb size (LABEL_NUM): {num_labels}")
    # --- ---

    # --- Calculate num_patents correctly (using absolute offsets for slicing later) 
    patent_start_idx_abs = label_offsets['patents'] # Absolute start index
    patent_end_idx_abs = num_labels + patent_start_idx_abs # Default if no subsequent offset
    if 'medium_cpcs' in label_offsets:
        patent_end_idx_abs = label_offsets['medium_cpcs']
    # ... (add other elif checks if needed) ...
    num_patents = patent_end_idx_abs - patent_start_idx_abs
    print(f"Derived num_patents: {num_patents} (Absolute index range {patent_start_idx_abs} to {patent_end_idx_abs-1})")
    

    X_figures_tensor = torch.tensor(X_figures, dtype=torch.float32)

    # ... (constraint tensor creation and chunking - uses relative indices now) ...
    # Ensure implication/exclusion tensors loaded from npz also use relative indices
    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    exclusion_tensor = torch.tensor(exclusion, dtype=torch.long, device=device) if exclusion else torch.empty((0, 2), dtype=torch.long, device=device)
    # --- Add validation for implication/exclusion relative indices ---
    if implication_tensor.numel() > 0:
        min_imp_idx = implication_tensor.min()
        max_imp_idx = implication_tensor.max()
        print(f"Validation: Implication indices range [{min_imp_idx.item()}, {max_imp_idx.item()}] vs Label Num {num_labels}")
        if min_imp_idx < 0 or max_imp_idx >= num_labels:
             print("ERROR: Invalid relative indices found in implication data!")
             # Handle error
    # --- ---

    # --- Prepare positive and negative pairs with CORRECTED VALIDATION ---\n    
    figure_to_pos_patent = {}
    figure_to_neg_patents = {}
    all_figure_indices_with_pairs = set()
    pos_pairs_processed = 0
    pos_pairs_skipped_fig_idx = 0
    pos_pairs_skipped_pat_idx = 0
    neg_pairs_processed = 0
    neg_pairs_skipped_fig_idx = 0
    neg_pairs_skipped_pat_idx = 0

    print("Processing loaded Y_pos...")
    for fig_idx, patent_idx_relative in Y_pos: # patent_idx_relative is now 0-based
        pos_pairs_processed += 1
        # Check figure index
        if not (0 <= fig_idx < num_figures):
            pos_pairs_skipped_fig_idx += 1
            continue
        # --- CORRECTED CHECK for relative patent index ---
        # Check if patent_idx_relative is within the valid range [0, num_labels - 1]
        if not (0 <= patent_idx_relative < num_labels):
            print(f"Warning: Invalid relative patent index {patent_idx_relative} (range 0-{num_labels-1}) found in Y_pos for figure {fig_idx}. Skipping.")
            pos_pairs_skipped_pat_idx += 1
            continue
        # --- Optional: Check if it falls within the expected patent sub-range ---
        # This requires knowing the relative start/end for patents
        patent_start_idx_rel = 0 # Patents are the first labels
        patent_end_idx_rel = num_patents
        if not (patent_start_idx_rel <= patent_idx_relative < patent_end_idx_rel):
             print(f"Note: Y_pos relative patent index {patent_idx_relative} is outside expected patent range [0-{patent_end_idx_rel-1}] but within label range [0-{num_labels-1}]. Accepting.")
             # Decide if this is acceptable or an error
        # --- ---

        figure_to_pos_patent[fig_idx] = patent_idx_relative # Store relative index
        all_figure_indices_with_pairs.add(fig_idx)

    print(f"Processed {pos_pairs_processed} positive pairs.")
    print(f"  Skipped {pos_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {pos_pairs_skipped_pat_idx} due to invalid relative patent index.")

    print("Processing loaded Y_neg...")
    for fig_idx, patent_idx_relative in Y_neg: # patent_idx_relative is now 0-based
        neg_pairs_processed += 1
        # Check figure index
        if not (0 <= fig_idx < num_figures):
            neg_pairs_skipped_fig_idx += 1
            continue
        # --- CORRECTED CHECK for relative patent index ---
        # Check if patent_idx_relative is within the valid range [0, num_labels - 1]
        if not (0 <= patent_idx_relative < num_labels):
            print(f"Warning: Invalid relative patent index {patent_idx_relative} (range 0-{num_labels-1}) found in Y_neg for figure {fig_idx}. Skipping.")
            neg_pairs_skipped_pat_idx += 1
            continue
        # --- Optional: Check if it falls within the expected patent sub-range ---
        patent_start_idx_rel = 0
        patent_end_idx_rel = num_patents
        if not (patent_start_idx_rel <= patent_idx_relative < patent_end_idx_rel):
             print(f"Note: Y_neg relative patent index {patent_idx_relative} is outside expected patent range [0-{patent_end_idx_rel-1}] but within label range [0-{num_labels-1}]. Accepting.")
             # Decide if this is acceptable or an error
        # --- ---

        if fig_idx not in figure_to_neg_patents:
            figure_to_neg_patents[fig_idx] = []
        figure_to_neg_patents[fig_idx].append(patent_idx_relative) # Store relative index

    print(f"Processed {neg_pairs_processed} negative pairs.")
    print(f"  Skipped {neg_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {neg_pairs_skipped_pat_idx} due to invalid relative patent index.")

    # --- Add diagnostic prints AFTER populating dicts ---
    print(f"Populated figure_to_pos_patent with {len(figure_to_pos_patent)} entries.")
    print(f"Populated figure_to_neg_patents with {len(figure_to_neg_patents)} entries.")
    if not figure_to_pos_patent or not figure_to_neg_patents:
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("ERROR: One or both mapping dictionaries are empty after processing loaded data. Check data preparation script and index ranges.")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         # Optionally exit or raise error here
         return model, -1.0 # Indicate error

    if figure_to_pos_patent:
        first_key = next(iter(figure_to_pos_patent))
        print(f"  Example figure_to_pos_patent entry: ({first_key}, {figure_to_pos_patent[first_key]})")
    if figure_to_neg_patents:
        first_key = next(iter(figure_to_neg_patents))
        num_negs_example = len(figure_to_neg_patents[first_key])
        print(f"  Example figure_to_neg_patents entry: ({first_key}, {figure_to_neg_patents[first_key][:5]}...) Num negs: {num_negs_example}")
        # Check distribution of negative counts
        neg_counts = [len(v) for v in figure_to_neg_patents.values()]
        print(f"  Negative sample counts per figure: Min={min(neg_counts)}, Max={max(neg_counts)}, Avg={np.mean(neg_counts):.2f}, Median={np.median(neg_counts)}")
        print(f"  Number of figures with < {num_neg_samples} negatives: {sum(1 for count in neg_counts if count < num_neg_samples)}")
    # --- End diagnostic prints ---

    # --- Check if any figures remain after filtering ---
    trainable_figure_indices = sorted(list(all_figure_indices_with_pairs))
    if not trainable_figure_indices:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: No valid figures found after processing Y_pos. Check data preparation and index validation.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return model, -1.0 # Indicate error
   

    # Convert features to tensor
    X_figures_tensor = torch.tensor(X_figures, dtype=torch.float32) # Keep on CPU initially

    # Convert constraint lists to tensors on the correct device
    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    exclusion_tensor = torch.tensor(exclusion, dtype=torch.long, device=device) if exclusion else torch.empty((0, 2), dtype=torch.long, device=device)

    # Chunk constraints
    implication_chunks = [implication_tensor[i:i+chunk_size] for i in range(0, implication_tensor.size(0), chunk_size)]
    exclusion_chunks = [exclusion_tensor[i:i+chunk_size] for i in range(0, exclusion_tensor.size(0), chunk_size)]
    print(f"Using {len(implication_chunks)} implication chunks and {len(exclusion_chunks)} exclusion chunks.")

    # Prepare positive and negative pairs efficiently
    # Group negatives by figure index for faster batch creation
    figure_to_pos_patent = {}
    figure_to_neg_patents = {}
    all_figure_indices_with_pairs = set()

    for fig_idx, patent_idx in Y_pos:
        figure_to_pos_patent[fig_idx] = patent_idx # Assume one positive patent per figure for simplicity here
        all_figure_indices_with_pairs.add(fig_idx)

    for fig_idx, patent_idx in Y_neg:
        if fig_idx not in figure_to_neg_patents:
            figure_to_neg_patents[fig_idx] = []
        figure_to_neg_patents[fig_idx].append(patent_idx)

    trainable_figure_indices = sorted(list(all_figure_indices_with_pairs))
    random.shuffle(trainable_figure_indices)

    # Split indices
    num_trainable = len(trainable_figure_indices)
    val_count = int(num_trainable * validation_split)
    test_count = int(num_trainable * test_split)
    train_count = num_trainable - val_count - test_count

    train_indices = trainable_figure_indices[:train_count]
    val_indices = trainable_figure_indices[train_count : train_count + val_count]
    test_indices = trainable_figure_indices[train_count + val_count :]

    print(f"Total figures with pairs: {num_trainable}")
    print(f"Training figures: {len(train_indices)}, Validation figures: {len(val_indices)}, Test figures: {len(test_indices)}")

    def create_triplet_batch(indices, batch_size, num_neg, figure_to_pos_patent, figure_to_neg_patents, X_figures_tensor, device): # Added more args
        print(f"Creating batches from {len(indices)} indices...")
        random.shuffle(indices)
        batches_yielded = 0 # Counter for yielded batches
        figures_processed = 0 # Counter for figures attempted

        for i in range(0, len(indices), batch_size):
            batch_fig_indices = indices[i : i+batch_size]
            figures_processed += len(batch_fig_indices)
            batch_pos_pat_indices = []
            batch_neg_pat_indices = []
            valid_batch_indices = [] # Keep track of figures for which we found pairs

            for fig_idx in batch_fig_indices:
                has_pos = fig_idx in figure_to_pos_patent
                has_negs = fig_idx in figure_to_neg_patents
                enough_negs = False
                if has_negs:
                    enough_negs = len(figure_to_neg_patents[fig_idx]) >= num_neg

                if has_pos and has_negs and enough_negs:
                    pos_pat = figure_to_pos_patent[fig_idx]
                    available_negs = figure_to_neg_patents[fig_idx]
                    neg_pats = random.sample(available_negs, num_neg)
                    batch_pos_pat_indices.append(pos_pat)
                    batch_neg_pat_indices.extend(neg_pats)
                    valid_batch_indices.append(fig_idx)
               
            if not valid_batch_indices:
                # print(f"Batch starting at index {i} is empty, skipping.") # Optional: Print empty batches
                continue # Skip if batch is empty

            # Get features for valid figures
            try:
                batch_x = X_figures_tensor[valid_batch_indices].to(device)
                batch_pos_pat = torch.tensor(batch_pos_pat_indices, dtype=torch.long, device=device)
                batch_neg_pat = torch.tensor(batch_neg_pat_indices, dtype=torch.long, device=device)

                #
                if batches_yielded == 0 and i == 0: # Print only for the very first batch
                    print(f"  First batch details: Num valid figures: {len(valid_batch_indices)}, "
                        f"X shape: {batch_x.shape}, Pos shape: {batch_pos_pat.shape}, Neg shape: {batch_neg_pat.shape}")
            

                yield batch_x, batch_pos_pat, batch_neg_pat
                batches_yielded += 1

            except IndexError as e:
                print(f"IndexError during batch creation: {e}. Valid indices: {valid_batch_indices[:5]}..., Max index in X: {X_figures_tensor.shape[0]-1}")
                continue # Skip this batch if indexing fails

        print(f"Finished batch creation: Yielded {batches_yielded} batches from {figures_processed} figures attempted.")
        if batches_yielded == 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: No batches were yielded. Check data pairing and negative sampling.")
            print(f"Example figure_to_pos_patent entry: {list(figure_to_pos_patent.items())[:1]}")
            print(f"Example figure_to_neg_patents entry: {list(figure_to_neg_patents.items())[:1]}")
            print(f"Num neg samples required: {num_neg}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # --- Optimizer and Early Stopping ---
    optimizer = gt.optim.RiemannianAdam(model.parameters(), lr=lr)
    # Use path for saving best model
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_val_map = -1.0 # Track best validation mAP

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_retrieval_loss = 0.0
        epoch_hier_loss = 0.0
        batch_count = 0
# --- Pass necessary args to batch creator ---
        batch_generator = create_triplet_batch(
            train_indices, batch_size, num_neg_samples,
            figure_to_pos_patent, figure_to_neg_patents, # Pass dicts
            X_figures_tensor, device # Pass features and device
        )
        # --- ---

        for batch_x, batch_pos_pat, batch_neg_pat in batch_generator: # Use the generatoroptimizer.zero_grad()

            # --- Forward Pass ---
            # Encode figures
            encoded_figures = model.encode_figures(batch_x)

            # Calculate hierarchical loss (process chunks) - only need label embeddings
            inside_loss_total = torch.tensor(0.0, device=device)
            disjoint_loss_total = torch.tensor(0.0, device=device)
            if implication_chunks or exclusion_chunks:
            
                 current_label_emb = model.label_emb # Use current embeddings
                 for imp_chunk in implication_chunks:
                     i_loss, _ = model.calculate_hierarchical_loss(imp_chunk, None)
                     inside_loss_total += i_loss
                 for exc_chunk in exclusion_chunks:
                     _, d_loss = model.calculate_hierarchical_loss(None, exc_chunk)
                     disjoint_loss_total += d_loss
                 # Average over chunks? Or sum? Summing might make penalty too large. Let's average.
                 if implication_chunks: inside_loss_total /= len(implication_chunks)
                 if exclusion_chunks: disjoint_loss_total /= len(exclusion_chunks)

            hierarchical_loss = inside_loss_total + disjoint_loss_total

            # Calculate regularization loss
            label_reg, instance_reg = model.calculate_reg_loss(encoded_figures)
            reg_loss = label_reg + instance_reg

            # Retrieval Loss 
            # Get corresponding label embeddings
            pos_label_emb = model.label_emb[batch_pos_pat]
            neg_label_emb = model.label_emb[batch_neg_pat]

            retrieval_loss = sample_to_prototype_loss(
                encoded_figures, pos_label_emb, neg_label_emb, model.k
            )

            # Total Loss 
            total_loss = (
                retrieval_loss
                + constraint_penalty * hierarchical_loss
                + reg_penalty * reg_loss
            )

       
            total_loss.backward()
         
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_retrieval_loss += retrieval_loss.item()
            epoch_hier_loss += hierarchical_loss.item() # Note: this is already averaged over chunks if applicable
            batch_count += 1

            if batch_count % 50 == 0:
                 print(f"Epoch {epoch}, Batch {batch_count}, Loss: {total_loss.item():.4f}, "
                       f"Retrieval: {retrieval_loss.item():.4f}, Hier: {hierarchical_loss.item():.4f}, Reg: {reg_loss.item():.4f}")
                 torch.cuda.empty_cache()

        avg_loss = epoch_total_loss / batch_count if batch_count > 0 else 0
        avg_retr_loss = epoch_retrieval_loss / batch_count if batch_count > 0 else 0
        avg_hier_loss = epoch_hier_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch} Summary - Avg Loss: {avg_loss:.4f}, Avg Retrieval: {avg_retr_loss:.4f}, Avg Hier: {avg_hier_loss:.4f}")

        # Validation 
        model.eval()
        val_map = evaluate_retrieval(model, X_figures_tensor, val_indices, figure_to_pos_patent, label_offsets, device, batch_size=batch_size*2) # Larger batch size for eval
        print(f"Epoch {epoch}, Validation mAP: {val_map:.4f}")

       
        if val_map > best_val_map:
            best_val_map = val_map
            print(f"New best validation mAP: {best_val_map:.4f}. Saving model...")
            torch.save(model.state_dict(), f'best_retrieval_model_c{model.c}_e{model.embed_dim}.pt')
            early_stopping.counter = 0 # Reset counter if performance improves
        else:
            early_stopping.counter += 1
            print(f'EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}')
            if early_stopping.counter >= early_stopping.patience:
                 print("Early stopping triggered.")
                 break

        torch.cuda.empty_cache()

    # --- Final Testing ---
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f'best_retrieval_model_c{model.c}_e{model.embed_dim}.pt'))
    model.eval()
    test_map = evaluate_retrieval(model, X_figures_tensor, test_indices, figure_to_pos_patent, label_offsets, device, batch_size=batch_size*2)
    print(f"Final Test mAP: {test_map:.4f}")

    return model, test_map

def hyperbolic_contrastive_loss(anchor_embeddings, positive_embeddings, k, temperature=0.07):
    """
    Compute hyperbolic contrastive loss (InfoNCE/NT-Xent) in hyperbolic space.
    
    Args:
        anchor_embeddings (torch.Tensor): Hyperbolic embeddings of anchor samples
        positive_embeddings (torch.Tensor): Hyperbolic embeddings of positive samples
        k (torch.Tensor): Curvature parameter of the hyperbolic space
        temperature (float): Temperature parameter for scaling distances
        
    Returns:
        torch.Tensor: Hyperbolic contrastive loss
    """
    n = anchor_embeddings.shape[0]
    device = anchor_embeddings.device
    
    # Calculate pairwise hyperbolic distances between all anchors and positives
    # Creating a matrix of shape [n, n] where each entry (i,j) is the distance
    # between anchor i and positive j
    all_distances = torch.zeros((n, n), device=device)
    
    for i in range(n):
        for j in range(n):
            # Calculate hyperbolic distance between anchor_i and positive_j
            dist = pmath.dist(
                anchor_embeddings[i].unsqueeze(0), 
                positive_embeddings[j].unsqueeze(0), 
                k=k
            ).squeeze()
            all_distances[i, j] = dist
    
    # Convert distances to similarities (smaller distance = higher similarity)
    # Negative sign because smaller distances should correspond to higher logits
    similarities = -all_distances / temperature
    
    # Labels are on the diagonal (each anchor matches with its corresponding positive)
    labels = torch.arange(n, device=device)
    
    # Compute cross-entropy loss in both directions (anchor→positive, positive→anchor)
    loss_a2p = F.cross_entropy(similarities, labels)
    loss_p2a = F.cross_entropy(similarities.t(), labels)
    
    # Average the two losses
    loss = (loss_a2p + loss_p2a) / 2
    
    return loss
def create_batch_with_figure_pairs(
    train_dataloader,
    num_neg,
    figure_to_pos_patent,
    figure_to_neg_patents,
    figure_to_pos_figures,
    figure_to_neg_figures,
    device
):
    """
    Yields batches combining:
      - anchor_images, positive_images (from the DataLoader)
      - concatenated images for CLIP/hyperbolic forward
      - pos_patents, neg_patents (LongTensors)
      - pos_fig_pairs, neg_fig_pairs (LongTensor pairs)
      - fig_indices (LongTensor)
    """
    import random
    for fig_indices, anchor_images, positive_images in train_dataloader:
        # Move to device
        fig_indices       = fig_indices.to(device)
        anchor_images     = anchor_images.to(device)
        positive_images   = positive_images.to(device)
        batch_size        = anchor_images.size(0)

        # Stack for a single forward pass
        images = torch.cat([anchor_images, positive_images], dim=0)

        # Build supervision lists
        pos_patents, neg_patents = [], []
        pos_fig_pairs, neg_fig_pairs = [], []

        for fig_idx in fig_indices.tolist():
            # — Patent positives / negatives —
            if fig_idx in figure_to_pos_patent and fig_idx in figure_to_neg_patents:
                pos_patents.append(figure_to_pos_patent[fig_idx])
                sampled_negs = random.sample(
                    figure_to_neg_patents[fig_idx],
                    min(len(figure_to_neg_patents[fig_idx]), num_neg)
                )
                neg_patents.extend(sampled_negs)
            else:
                # fallback to self (or pad)
                pos_patents.append(fig_idx)
                neg_patents.extend([fig_idx] * num_neg)

            # — Figure‐to‐figure pairs —
            pos_peer = (
                random.choice(figure_to_pos_figures[fig_idx])
                if fig_idx in figure_to_pos_figures and figure_to_pos_figures[fig_idx]
                else fig_idx
            )
            neg_peer = (
                random.choice(figure_to_neg_figures[fig_idx])
                if fig_idx in figure_to_neg_figures and figure_to_neg_figures[fig_idx]
                else fig_idx
            )
            pos_fig_pairs.append((fig_idx, pos_peer))
            neg_fig_pairs.append((fig_idx, neg_peer))

        
        pos_patents   = torch.tensor(pos_patents,   dtype=torch.long, device=device)
        neg_patents   = torch.tensor(neg_patents,   dtype=torch.long, device=device)
        pos_fig_pairs = torch.tensor(pos_fig_pairs, dtype=torch.long, device=device)
        neg_fig_pairs = torch.tensor(neg_fig_pairs, dtype=torch.long, device=device)

        yield {
            "fig_indices":      fig_indices,
            "anchor_images":    anchor_images,
            "positive_images":  positive_images,
            "images":           images,
            "pos_patents":      pos_patents,
            "neg_patents":      neg_patents,
            "pos_fig_pairs":    pos_fig_pairs,
            "neg_fig_pairs":    neg_fig_pairs,
        }

    
def train_end_to_end_old(
    clip_model,
    hyperbolic_model,
    train_dataloader,
    val_dataloader,
    Y_pos,
    Y_neg,
    implication,
    exclusion,
    label_offsets,
    positive_figure_pairs,  # New: List of (figure_idx1, figure_idx2) from same patent
    negative_figure_pairs,

    epochs=10,
    clip_lr=2e-5,
    hyperbolic_lr=1e-3,
    temperature=0.07,
    device=None,
    save_dir="models",
    patience=5,
    clip_finetune=True,  # Whether to fine-tune CLIP or keep it frozen
    clip_weight=1.0,     # Weight for CLIP loss
    hyperbolic_weight=1.0,  # Weight for hyperbolic loss
    use_wandb=True,      # Whether to use Weights & Biases
    wandb_project="hyperbolic-clip-end2end",  # W&B project name
    wandb_run_name=None,

    
    num_neg_samples=1, 
    figure_pair_weight=2,
    constraint_penalty=3,
    retrieval_penalty=2, 
    reg_penalty=0.01,
):
    """
    End-to-end training of CLIP and hyperbolic embedding models with W&B tracking
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to device
    clip_model = clip_model.to(device)
    hyperbolic_model = hyperbolic_model.to(device)
    # Freeze base model and unfreeze specific layers
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers of vision encoder
    for param in clip_model.vision_model.encoder.layers[-9:].parameters():
        param.requires_grad = True
    
    #for param in hyperbolic_model.encoder.first_layer.parameters(): 
    #    param.requires_grad = False

    #for name, param in hyperbolic_model.named_parameters():
    #    print(f"{name}: requires_grad={param.requires_grad}")


    hyperbolic_model.k = hyperbolic_model.k.to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Weights & Biases
    if use_wandb:
        import wandb
        config = {
            "clip_lr": clip_lr,
            "hyperbolic_lr": hyperbolic_lr,
            "temperature": temperature,
            "clip_finetune": clip_finetune,
            "clip_weight": clip_weight,
            "hyperbolic_weight": hyperbolic_weight,
            "patience": patience,
            "device": str(device)
        }
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        # Log model architecture (optional)
        wandb.watch(hyperbolic_model)
        if clip_finetune:
            wandb.watch(clip_model)
    
    # --- Data Preparation ---
    num_figures = len(Y_pos)
    num_labels = hyperbolic_model.label_emb.shape[0]
    print(f"Model label_emb size (LABEL_NUM): {num_labels}")
    print(f"Model num_figures size : {num_figures}")
   
    patent_start_idx_abs = label_offsets['patents']  # Absolute start index
    patent_end_idx_abs = num_labels + patent_start_idx_abs  # Default if no subsequent offset
    if 'medium_cpcs' in label_offsets:
        patent_end_idx_abs = label_offsets['medium_cpcs']
    num_patents = patent_end_idx_abs - patent_start_idx_abs
    print(f"Derived num_patents: {num_patents} (Absolute index range {patent_start_idx_abs} to {patent_end_idx_abs-1})")
    
    # Process figure-to-figure pairs 
    has_figure_pairs = (positive_figure_pairs is not None and len(positive_figure_pairs) > 0) and \
                       (negative_figure_pairs is not None and len(negative_figure_pairs) > 0)
    
    if has_figure_pairs:
        print(f"Processing {len(positive_figure_pairs)} positive and {len(negative_figure_pairs)} negative figure pairs...")
        # Convert to tensors
        pos_fig_pairs_tensor = torch.tensor(positive_figure_pairs, dtype=torch.long)
        neg_fig_pairs_tensor = torch.tensor(negative_figure_pairs, dtype=torch.long)
        
        # Validate figure indices
        pos_fig_pairs_valid = ((0 <= pos_fig_pairs_tensor) & (pos_fig_pairs_tensor < num_figures)).all()
        neg_fig_pairs_valid = ((0 <= neg_fig_pairs_tensor) & (neg_fig_pairs_tensor < num_figures)).all()
        
        if not pos_fig_pairs_valid:
            print("WARNING: Some positive figure pair indices are out of bounds. Filtering...")
            valid_mask = ((0 <= pos_fig_pairs_tensor[:, 0]) & (pos_fig_pairs_tensor[:, 0] < num_figures) & 
                          (0 <= pos_fig_pairs_tensor[:, 1]) & (pos_fig_pairs_tensor[:, 1] < num_figures))
            pos_fig_pairs_tensor = pos_fig_pairs_tensor[valid_mask]
            
        if not neg_fig_pairs_valid:
            print("WARNING: Some negative figure pair indices are out of bounds. Filtering...")
            valid_mask = ((0 <= neg_fig_pairs_tensor[:, 0]) & (neg_fig_pairs_tensor[:, 0] < num_figures) & 
                          (0 <= neg_fig_pairs_tensor[:, 1]) & (neg_fig_pairs_tensor[:, 1] < num_figures))
            neg_fig_pairs_tensor = neg_fig_pairs_tensor[valid_mask]
        
        print(f"After validation: {len(pos_fig_pairs_tensor)} positive and {len(neg_fig_pairs_tensor)} negative figure pairs")
        
        # Group figure pairs by first index for batch processing
        figure_to_pos_figures = defaultdict(list)
        figure_to_neg_figures = defaultdict(list)
        
        for fig1, fig2 in pos_fig_pairs_tensor.tolist():
            figure_to_pos_figures[fig1].append(fig2)
            # Also add the reverse pair for symmetry
            figure_to_pos_figures[fig2].append(fig1)
            
        for fig1, fig2 in neg_fig_pairs_tensor.tolist():
            figure_to_neg_figures[fig1].append(fig2)
            # Also add the reverse pair for symmetry
            figure_to_neg_figures[fig2].append(fig1)
            
        print(f"Grouped figure pairs: {len(figure_to_pos_figures)} figures with positive pairs, "
              f"{len(figure_to_neg_figures)} figures with negative pairs")
    else:
        print("No figure-to-figure pairs provided. Skipping figure pair loss.")
        figure_to_pos_figures = {}
        figure_to_neg_figures = {}



    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    exclusion_tensor = torch.tensor(exclusion, dtype=torch.long, device=device) if exclusion else torch.empty((0, 2), dtype=torch.long, device=device)

    figure_to_pos_patent = {}
    figure_to_neg_patents = {}
    all_figure_indices_with_pairs = set()
    pos_pairs_processed = 0
    pos_pairs_skipped_fig_idx = 0
    pos_pairs_skipped_pat_idx = 0
    neg_pairs_processed = 0
    neg_pairs_skipped_fig_idx = 0
    neg_pairs_skipped_pat_idx = 0

    print("Processing loaded Y_pos...")
    for fig_idx, patent_idx_relative in Y_pos:  # patent_idx_relative is now 0-based
        pos_pairs_processed += 1
        # Check figure index
        if not (0 <= fig_idx < num_figures):
            pos_pairs_skipped_fig_idx += 1
            continue

        if not (0 <= patent_idx_relative < num_labels):
            print(f"Warning: Invalid relative patent index {patent_idx_relative} (range 0-{num_labels-1}) found in Y_pos for figure {fig_idx}. Skipping.")
            pos_pairs_skipped_pat_idx += 1
            continue

        patent_start_idx_rel = 0  # Patents are the first labels
        patent_end_idx_rel = num_patents
        if not (patent_start_idx_rel <= patent_idx_relative < patent_end_idx_rel):
             print(f"Note: Y_pos relative patent index {patent_idx_relative} is outside expected patent range [0-{patent_end_idx_rel-1}] but within label range [0-{num_labels-1}]. Accepting.")

        figure_to_pos_patent[fig_idx] = patent_idx_relative  # Store relative index
        all_figure_indices_with_pairs.add(fig_idx)

    print(f"Processed {pos_pairs_processed} positive pairs.")
    print(f"  Skipped {pos_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {pos_pairs_skipped_pat_idx} due to invalid relative patent index.")

    print("Processing loaded Y_neg...")
    for fig_idx, patent_idx_relative in Y_neg:  # patent_idx_relative is now 0-based
        neg_pairs_processed += 1
        # Check figure index
   
        if fig_idx not in figure_to_neg_patents:
            figure_to_neg_patents[fig_idx] = []
        figure_to_neg_patents[fig_idx].append(patent_idx_relative)  # Store relative index

    print(f"Processed {neg_pairs_processed} negative pairs.")
    print(f"  Skipped {neg_pairs_skipped_fig_idx} due to invalid figure index.")
    print(f"  Skipped {neg_pairs_skipped_pat_idx} due to invalid relative patent index.")

    
    print(f"Populated figure_to_pos_patent with {len(figure_to_pos_patent)} entries.")
    print(f"Populated figure_to_neg_patents with {len(figure_to_neg_patents)} entries.")
    


    # Convert constraint lists to tensors on the correct device
    implication_tensor = torch.tensor(implication, dtype=torch.long, device=device) if implication else torch.empty((0, 2), dtype=torch.long, device=device)
    
    
    trainable_figure_indices = sorted(list(all_figure_indices_with_pairs))
    random.shuffle(trainable_figure_indices)
    # Split indices
    #num_trainable = len(trainable_figure_indices)
    #val_count = int(num_trainable * validation_split)
    #test_count = int(num_trainable * test_split)
    #train_count = num_trainable - val_count - test_count

    #train_indices = trainable_figure_indices[:train_count]
    #val_indices = trainable_figure_indices[train_count : train_count + val_count]
    #test_indices = trainable_figure_indices[train_count + val_count :]

    #print(f"Total figures with pairs: {num_trainable}")
    #print(f"Training figures: {len(train_indices)}, Validation figures: {len(val_indices)}, Test figures: {len(test_indices)}")
    manifold_params = [p for n, p in hyperbolic_model.named_parameters() if "label_emb" in n]
    euclidean_params = [p for n, p in hyperbolic_model.named_parameters() if "label_emb" not in n]

    
    
    clip_optimizer = torch.optim.AdamW(clip_model.parameters(), lr=clip_lr)
    hyperbolic_optimizer = torch.optim.Adam(euclidean_params, lr=hyperbolic_lr)
    hyperbolic_optimizer_2 = RiemannianAdam(manifold_params, lr=hyperbolic_lr)
    # Set up early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    
    


    # Training loop
    for epoch in range(1, epochs + 1):
        # --- Training ---
        clip_model.train()
        hyperbolic_model.train()
        
        total_loss = 0.0
        epoch_clip_loss = 0.0
        epoch_hyperbolic_loss = 0.0
        batch_count = 0

        epoch_total_loss = 0.0
        epoch_retrieval_loss = 0.0
        epoch_hier_loss = 0.0
        epoch_reg_loss=0.0
        epoch_figure_pair_loss = 0.0
        batch_count = 0

        batch_generator = create_batch_with_figure_pairs(
        train_dataloader,
        num_neg=1,
        figure_to_pos_patent=figure_to_pos_patent,
        figure_to_neg_patents=figure_to_neg_patents,
        figure_to_pos_figures=figure_to_pos_figures,
        figure_to_neg_figures=figure_to_neg_figures,
        device=device
        )
        for batch in tqdm(batch_generator, desc=f"Epoch {epoch} Training"):
            #if batch is None:
                #continue
            images_ = batch["images"]
            a_imgs = batch["anchor_images"]
            p_imgs = batch["positive_images"]
            batch_pos_pat = batch["pos_patents"]
            batch_neg_pat = batch["neg_patents"]
            batch_pos_fig_pairs = batch["pos_fig_pairs"]
            batch_neg_fig_pairs = batch["neg_fig_pairs"]   
            batch_pos_fig_pairs = torch.tensor(batch_pos_fig_pairs, dtype=torch.long, device=device)
            batch_neg_fig_pairs = torch.tensor(batch_neg_fig_pairs, dtype=torch.long, device=device)
            
            images = torch.cat([a_imgs, p_imgs], dim=0)
            images = images.view(-1, 3, 224, 224).to(device)
            n = a_imgs.shape[0]
            
            
            # --- Forward pass through CLIP ---

            with torch.set_grad_enabled(clip_finetune):
                # Get CLIP image features
                clip_features = clip_model.get_image_features(pixel_values=images)
                
                # Normalize features
                clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                
                # Split features
                anchor_clip_features = clip_features[:n]
                positive_clip_features = clip_features[n:]
                
                # Compute CLIP contrastive loss (cosine similarity based)
                logits_per_image = torch.matmul(anchor_clip_features, positive_clip_features.t()) / temperature
                labels = torch.arange(n, device=device)
                clip_loss = (F.cross_entropy(logits_per_image, labels) + 
                             F.cross_entropy(logits_per_image.t(), labels)) / 2
            
            # --- Forward pass through Hyperbolic ---
            
            # Encode CLIP features into hyperbolic space
            hyperbolic_embeddings = hyperbolic_model(clip_features)
            
            # Split hyperbolic embeddings
            anchor_hyperbolic = hyperbolic_embeddings[:n]
            positive_hyperbolic = hyperbolic_embeddings[n:]

            

            # Calculate hierarchical loss (process chunks)
            inside_loss_total = torch.tensor(0.0, device=device)
            disjoint_loss_total = torch.tensor(0.0, device=device)
            
            current_label_emb = hyperbolic_model.label_emb
            hierarchical_loss, _ = hyperbolic_model.calculate_hierarchical_loss(implication_tensor, None)

            # Calculate regularization loss
            label_reg, instance_reg = hyperbolic_model.calculate_reg_loss(anchor_hyperbolic)
            reg_loss = label_reg + instance_reg

            # --- Retrieval Loss ---
            # Get corresponding label embeddings
            pos_label_emb = hyperbolic_model.label_emb[batch_pos_pat]
            neg_label_emb = hyperbolic_model.label_emb[batch_neg_pat]

            retrieval_loss = sample_to_prototype_loss(
                anchor_hyperbolic, pos_label_emb, neg_label_emb,num_neg_samples, hyperbolic_model.k
            )

            # Compute hyperbolic contrastive loss
            hyperbolic_cross_loss = hyperbolic_contrastive_loss(
                anchor_hyperbolic, positive_hyperbolic, 
                k=hyperbolic_model.k, temperature=temperature
            )
            
            hyperbolic_loss=(
                retrieval_penalty+ retrieval_loss
                + constraint_penalty * hierarchical_loss
                + reg_penalty * reg_loss
                + figure_pair_weight * hyperbolic_cross_loss
            )

            # Combine losses with weights
            loss = clip_weight * clip_loss + (1-clip_weight) * hyperbolic_loss

            # Backward and optimize
            clip_optimizer.zero_grad()
            hyperbolic_optimizer.zero_grad()
            hyperbolic_optimizer_2.zero_grad()

            loss.backward()

            clip_optimizer.step()
            hyperbolic_optimizer.step()
            hyperbolic_optimizer_2.step()
                        
            # Track losses
            
            epoch_total_loss += loss.item()
            epoch_clip_loss += clip_loss.item()
            epoch_hyperbolic_loss += hyperbolic_loss.item()
            epoch_retrieval_loss += retrieval_loss.item()
            epoch_hier_loss += hierarchical_loss.item()
            epoch_reg_loss+= reg_loss.item()
            epoch_figure_pair_loss += hyperbolic_cross_loss.item()
            batch_count += 1
            
            # Print progress
            if batch_count % 10 == 0:
                print(f"Batch {batch_count}, Loss: {loss.item():.4f}, "
                    f"CLIP Loss: {clip_loss.item():.4f}, "
                      f"Hyperbolic Loss: {hyperbolic_loss.item():.4f}, "
                      f"Retrieval: {retrieval_loss.item():.4f}, Hier: {hierarchical_loss.item():.4f}, "
                      f"Hyperbolic cross-entropy: {hyperbolic_cross_loss.item():.4f}, Reg: {reg_loss.item():.4f}")      
            

            if use_wandb and batch_count % 10 == 0:
                wandb.log({
                    "batch": batch_count + (epoch-1) *128,
                    "batch_loss": loss.item(),
                    "batch_clip_loss": clip_loss.item(),
                    "batch_hyperbolic_loss": hyperbolic_loss.item(),
                    "retrieval_loss": retrieval_loss.item(),
                    "hierarchical_loss": hierarchical_loss.item(),
                    "Hyperbolic cross-entropy": hyperbolic_cross_loss.item(),
                    "reg_loss" : reg_loss.item()
                })
        # Calculate average losses
            if batch_count % 30 == 0:
                # --- Validation ---
                clip_model.eval()
                hyperbolic_model.eval()
                
                val_total_loss = 0.0
                val_clip_loss_sum = 0.0
                val_hyperbolic_loss_sum = 0.0
                val_retrieval_loss = 0.0
                val_hier_loss = 0.0
                val_reg_loss = 0.0
                val_fig_pair_loss = 0.0
                val_batch_count = 0

                val_batch_generator = create_batch_with_figure_pairs(
                    val_dataloader,
                    num_neg=1,
                    figure_to_pos_patent=figure_to_pos_patent,
                    figure_to_neg_patents=figure_to_neg_patents,
                    figure_to_pos_figures=figure_to_pos_figures,
                    figure_to_neg_figures=figure_to_neg_figures,
                    device=device
                )

                with torch.no_grad():
                    for batch in tqdm(val_batch_generator, desc=f"Epoch {epoch} Validation"):
                        images = batch["images"]
                        a_imgs = batch["anchor_images"]
                        p_imgs = batch["positive_images"]
                        batch_pos_pat = batch["pos_patents"]
                        batch_neg_pat = batch["neg_patents"]
                        batch_pos_fig_pairs = batch["pos_fig_pairs"]
                        batch_neg_fig_pairs = batch["neg_fig_pairs"]
                        n = a_imgs.shape[0]

                        # Forward pass through CLIP
                        clip_features = clip_model.get_image_features(pixel_values=images)
                        clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                        anchor_clip_features = clip_features[:n]
                        positive_clip_features = clip_features[n:]
                        logits_per_image = torch.matmul(anchor_clip_features, positive_clip_features.t()) / temperature
                        labels = torch.arange(n, device=device)
                        val_clip_loss = (F.cross_entropy(logits_per_image, labels) + 
                                        F.cross_entropy(logits_per_image.t(), labels)) / 2

                        # Forward through hyperbolic model
                        hyperbolic_embeddings = hyperbolic_model(clip_features)
                        anchor_hyp = hyperbolic_embeddings[:n]
                        positive_hyp = hyperbolic_embeddings[n:]

                        val_hierarchical_loss, _ = hyperbolic_model.calculate_hierarchical_loss(implication_tensor, None)
                        label_reg, instance_reg = hyperbolic_model.calculate_reg_loss(anchor_hyp)
                        val_reg_loss = label_reg + instance_reg

                        pos_label_emb = hyperbolic_model.label_emb[batch_pos_pat]
                        neg_label_emb = hyperbolic_model.label_emb[batch_neg_pat]
                        val_retrieval_loss = sample_to_prototype_loss(
                            anchor_hyp, pos_label_emb, neg_label_emb, num_neg_samples, hyperbolic_model.k)

                        val_figure_pair_loss = hyperbolic_contrastive_loss(
                            anchor_hyp, positive_hyp, k=hyperbolic_model.k, temperature=temperature)

                        val_hyperbolic_loss = (
                            retrieval_penalty + val_retrieval_loss +
                            constraint_penalty * val_hierarchical_loss +
                            reg_penalty * val_reg_loss 
                            +figure_pair_weight * val_figure_pair_loss
                        )

                        val_loss = clip_weight * val_clip_loss + hyperbolic_weight * val_hyperbolic_loss

                        val_total_loss += val_loss.item()
                        val_clip_loss_sum += val_clip_loss.item()
                        val_hyperbolic_loss_sum += val_hyperbolic_loss.item()
                        val_retrieval_loss += val_retrieval_loss.item()
                        val_hier_loss += val_hierarchical_loss.item()
                        val_reg_loss += val_reg_loss.item()
                        val_fig_pair_loss += val_figure_pair_loss.item()
                        val_batch_count += 1

                avg_val_loss = val_total_loss / val_batch_count
                avg_val_clip_loss = val_clip_loss_sum / val_batch_count
                avg_val_hyperbolic_loss = val_hyperbolic_loss_sum / val_batch_count
                avg_val_retr_loss = val_retrieval_loss / val_batch_count
                avg_val_hier_loss = val_hier_loss / val_batch_count
                avg_val_reg_loss = val_reg_loss / val_batch_count
                avg_val_fig_pair_loss = val_fig_pair_loss / val_batch_count

                print(f"Epoch {epoch} Validation - Avg Loss: {avg_val_loss:.4f}, Clip: {avg_val_clip_loss:.4f}, Hyp: {avg_val_hyperbolic_loss:.4f}, "
                    f"Retrieval: {avg_val_retr_loss:.4f}, Hier: {avg_val_hier_loss:.4f}, Reg: {avg_val_reg_loss:.4f}, Pair: {avg_val_fig_pair_loss:.4f}")

                # Log epoch metrics to W&B
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val_loss": avg_val_loss,
                        "val_clip_loss": avg_val_clip_loss,
                        "val_hyperbolic_loss": avg_val_hyperbolic_loss,
                        "learning_rate_clip": clip_lr if clip_finetune else 0,
                        "learning_rate_hyperbolic": hyperbolic_lr
                    })
                
                # Save models and check for early stopping
                if avg_val_loss < best_val_loss:
                    print(f"New best validation loss: {avg_val_loss:.4f}. Saving models...")
                    
                    # Save CLIP model
                    if clip_finetune:
        
                        # Also save best model
                        clip_save_path = os.path.join(save_dir, "clip_model_e2e_full_256_2_.pkl")
                        with open(clip_save_path, 'wb') as f:
                            pickle.dump(clip_model.state_dict(), f)
                        
                        if use_wandb:
                            wandb.save(clip_save_path)
                    
                    # Save hyperbolic model
                    hyperbolic_save_path = os.path.join(save_dir, f"hyperbolic_model_e2e_full_256_2_.pt")
                    torch.save(hyperbolic_model.state_dict(), hyperbolic_save_path)
                    
                    
                    if use_wandb:
                        wandb.save(hyperbolic_save_path)
                    
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"EarlyStopping counter: {patience_counter} out of {patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
    
        
        
        avg_loss = epoch_total_loss / batch_count if batch_count > 0 else 0
        avg_clip_loss = epoch_clip_loss / batch_count if batch_count > 0 else 0
        avg_hyperbolic_loss = epoch_hyperbolic_loss / batch_count if batch_count > 0 else 0
        avg_retr_loss = epoch_retrieval_loss / batch_count if batch_count > 0 else 0
        avg_hier_loss = epoch_hier_loss / batch_count if batch_count > 0 else 0
        avg_fig_pair_loss = epoch_figure_pair_loss / batch_count if batch_count > 0 else 0
        avg_reg_loss = epoch_reg_loss/ batch_count if batch_count > 0 else 0

        print(f"Epoch {epoch} Summary - Avg Loss: {avg_loss:.4f} ,Avg CLIP Loss: {avg_clip_loss:.4f}, Avg HYPER Loss: {avg_hyperbolic_loss:.4f}   Avg Retrieval: {avg_retr_loss:.4f}, Avg Hier: {avg_hier_loss:.4f}, Avg Fig Pair: {avg_fig_pair_loss:.4f}, Avg Reg: {avg_reg_loss:.4f}")
        
        
        # --- Validation ---
        clip_model.eval()
        hyperbolic_model.eval()
        
        val_total_loss = 0.0
        val_clip_loss_sum = 0.0
        val_hyperbolic_loss_sum = 0.0
        val_retrieval_loss = 0.0
        val_hier_loss = 0.0
        val_reg_loss = 0.0
        val_fig_pair_loss = 0.0
        val_batch_count = 0

        val_batch_generator = create_batch_with_figure_pairs(
            val_dataloader,
            num_neg=1,
            figure_to_pos_patent=figure_to_pos_patent,
            figure_to_neg_patents=figure_to_neg_patents,
            figure_to_pos_figures=figure_to_pos_figures,
            figure_to_neg_figures=figure_to_neg_figures,
            device=device
        )

        with torch.no_grad():
            for batch in tqdm(val_batch_generator, desc=f"Epoch {epoch} Validation"):
                images = batch["images"]
                a_imgs = batch["anchor_images"]
                p_imgs = batch["positive_images"]
                batch_pos_pat = batch["pos_patents"]
                batch_neg_pat = batch["neg_patents"]
                batch_pos_fig_pairs = batch["pos_fig_pairs"]
                batch_neg_fig_pairs = batch["neg_fig_pairs"]
                n = a_imgs.shape[0]

                # Forward pass through CLIP
                clip_features = clip_model.get_image_features(pixel_values=images)
                clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                anchor_clip_features = clip_features[:n]
                positive_clip_features = clip_features[n:]
                logits_per_image = torch.matmul(anchor_clip_features, positive_clip_features.t()) / temperature
                labels = torch.arange(n, device=device)
                val_clip_loss = (F.cross_entropy(logits_per_image, labels) + 
                                F.cross_entropy(logits_per_image.t(), labels)) / 2

                # Forward through hyperbolic model
                hyperbolic_embeddings = hyperbolic_model(clip_features)
                anchor_hyp = hyperbolic_embeddings[:n]
                positive_hyp = hyperbolic_embeddings[n:]

                val_hierarchical_loss, _ = hyperbolic_model.calculate_hierarchical_loss(implication_tensor, None)
                label_reg, instance_reg = hyperbolic_model.calculate_reg_loss(anchor_hyp)
                val_reg_loss = label_reg + instance_reg

                pos_label_emb = hyperbolic_model.label_emb[batch_pos_pat]
                neg_label_emb = hyperbolic_model.label_emb[batch_neg_pat]
                val_retrieval_loss = sample_to_prototype_loss(
                    anchor_hyp, pos_label_emb, neg_label_emb, num_neg_samples, hyperbolic_model.k)

                val_figure_pair_loss = hyperbolic_contrastive_loss(
                    anchor_hyp, positive_hyp, k=hyperbolic_model.k, temperature=temperature)

                val_hyperbolic_loss = (
                    retrieval_penalty + val_retrieval_loss +
                    constraint_penalty * val_hierarchical_loss +
                    reg_penalty * val_reg_loss 
                    +figure_pair_weight * val_figure_pair_loss
                )

                val_loss = clip_weight * val_clip_loss + hyperbolic_weight * val_hyperbolic_loss

                val_total_loss += val_loss.item()
                val_clip_loss_sum += val_clip_loss.item()
                val_hyperbolic_loss_sum += val_hyperbolic_loss.item()
                val_retrieval_loss += val_retrieval_loss.item()
                val_hier_loss += val_hierarchical_loss.item()
                val_reg_loss += val_reg_loss.item()
                val_fig_pair_loss += val_figure_pair_loss.item()
                val_batch_count += 1

        avg_val_loss = val_total_loss / val_batch_count
        avg_val_clip_loss = val_clip_loss_sum / val_batch_count
        avg_val_hyperbolic_loss = val_hyperbolic_loss_sum / val_batch_count
        avg_val_retr_loss = val_retrieval_loss / val_batch_count
        avg_val_hier_loss = val_hier_loss / val_batch_count
        avg_val_reg_loss = val_reg_loss / val_batch_count
        avg_val_fig_pair_loss = val_fig_pair_loss / val_batch_count

        print(f"Epoch {epoch} Validation - Avg Loss: {avg_val_loss:.4f}, Clip: {avg_val_clip_loss:.4f}, Hyp: {avg_val_hyperbolic_loss:.4f}, "
            f"Retrieval: {avg_val_retr_loss:.4f}, Hier: {avg_val_hier_loss:.4f}, Reg: {avg_val_reg_loss:.4f}, Pair: {avg_val_fig_pair_loss:.4f}")

        # Log epoch metrics to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_clip_loss": avg_clip_loss,
                "train_hyperbolic_loss": avg_hyperbolic_loss,
                "val_loss": avg_val_loss,
                "val_clip_loss": avg_val_clip_loss,
                "val_hyperbolic_loss": avg_val_hyperbolic_loss,
                "learning_rate_clip": clip_lr if clip_finetune else 0,
                "learning_rate_hyperbolic": hyperbolic_lr
            })
        
        # Save models and check for early stopping
        if avg_val_loss < best_val_loss:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving models...")
            
            # Save CLIP model
            if clip_finetune:
 
                # Also save best model
                clip_save_path = os.path.join(save_dir, "clip_model_e2e_full_256_2_.pkl")
                with open(clip_save_path, 'wb') as f:
                    pickle.dump(clip_model.state_dict(), f)
                
                if use_wandb:
                    wandb.save(clip_save_path)
            
            # Save hyperbolic model
            hyperbolic_save_path = os.path.join(save_dir, f"hyperbolic_model_e2e_full_256_2_.pt")
            torch.save(hyperbolic_model.state_dict(), hyperbolic_save_path)
            
            
            if use_wandb:
                wandb.save(hyperbolic_save_path)
            
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best models
    best_hyperbolic_path = os.path.join(save_dir, "hyperbolic_model_e2e_full_256_2_.pt")
    if os.path.exists(best_hyperbolic_path):
        hyperbolic_model.load_state_dict(torch.load(best_hyperbolic_path))
        print(f"Loaded best hyperbolic model from {best_hyperbolic_path}")
    
    if clip_finetune:
        best_clip_path = os.path.join(save_dir, "hyperbolic_model_e2e_full_256_2_.pkl")
        if os.path.exists(best_clip_path):
            with open(best_clip_path, 'rb') as f:
                clip_model.load_state_dict(pickle.load(f))
            print(f"Loaded best CLIP model from {best_clip_path}")
    
    # Finish W&B run
    
        
        wandb.finish()
    
    return clip_model, hyperbolic_model

def evaluate_retrieval(model, X_figures_tensor, eval_indices, figure_to_pos_patent, label_offsets, device, batch_size):
    """ 
    Evaluates retrieval performance using mAP.
    Now supports multiple positive patents per figure.
    """
    print(f"Evaluating retrieval on {len(eval_indices)} indices...")
    if not eval_indices:  # Handle empty eval set
        print("Evaluation set is empty.")
        return 0.0
    model.eval()

    # --- Setup label embeddings and patent slice ---
    all_label_emb = model.label_emb.detach()
    num_labels = all_label_emb.shape[0]
    patent_start_idx_abs = label_offsets.get('patents', -1)
    if patent_start_idx_abs == -1:
        print("  Error: 'patents' offset not found in label_offsets.")
        return -1.0

    # Calculate num_patents based on offsets
    num_patents = 0
    patent_end_idx_abs = num_labels + patent_start_idx_abs  # Default assumption
    # Find the next offset to determine the end
    potential_next_offsets = [v for k, v in label_offsets.items() if v > patent_start_idx_abs]
    if potential_next_offsets:
        patent_end_idx_abs = min(potential_next_offsets)

    num_patents = patent_end_idx_abs - patent_start_idx_abs

    patent_start_idx_rel = 0  # Patents start at relative index 0 in the label_emb tensor if they are the first type
    patent_end_idx_rel = num_patents
    all_patent_emb = all_label_emb[patent_start_idx_rel:patent_end_idx_rel]

    print(f"  Using patent embeddings relative indices {patent_start_idx_rel} to {patent_end_idx_rel-1} (Count: {num_patents})")
    if num_patents <= 0:  # Check for non-positive count
        print(f"  Error: Number of patent embeddings for evaluation is {num_patents}. Check label_offsets and calculation.")
        return 0.0
    # --- ---

    figure_embeddings_list = []  # Store encoded batches here
    encoded_count = 0
    # --- Encode figures in batches ---
    print(f"  Starting encoding for {len(eval_indices)} figures...")
    with torch.no_grad():
        for i in range(0, len(eval_indices), batch_size):
            batch_indices = eval_indices[i : i+batch_size]
            if not batch_indices:
                print(f"  Skipping empty batch slice at index {i}")
                continue

            try:
                # Ensure batch_indices are valid for X_figures_tensor
                max_index_in_batch = max(batch_indices)
                if max_index_in_batch >= X_figures_tensor.shape[0]:
                     print(f"    Error: Index {max_index_in_batch} in batch_indices is out of bounds for X_figures_tensor (size {X_figures_tensor.shape[0]}). Skipping batch.")
                     continue

                batch_x = X_figures_tensor[batch_indices].to(device)
                if batch_x.shape[0] == 0:
                    print(f"    Warning: batch_x is empty for indices {batch_indices[:5]}... Skipping batch.")
                    continue

                encoded = model.encode_figures(batch_x)

                if encoded.shape[0] != len(batch_indices):
                     print(f"    Warning: Encoder returned unexpected shape {encoded.shape} for batch size {len(batch_indices)}. Skipping batch.")
                     continue

                figure_embeddings_list.append(encoded.cpu())  # Move to CPU
                encoded_count += encoded.shape[0]

            except IndexError as e:
                 print(f"  IndexError during evaluation encoding: {e}. Indices: {batch_indices[:5]}..., Max index in X: {X_figures_tensor.shape[0]-1}")
                 print(f"  Skipping evaluation due to encoding error.")
                 return -1.0  # Indicate error
            except Exception as e:  # Catch other potential errors during encoding
                 print(f"  Unexpected error during encoding batch {i//batch_size + 1}: {e}")
                 print(f"  Skipping evaluation due to encoding error.")
                 return -1.0

    print(f"  Finished encoding. Total figures successfully encoded: {encoded_count}")

    if not figure_embeddings_list:
         print("  Error: No figure embeddings generated during evaluation (list is empty).")
         return 0.0

    try:
        figure_embeddings = torch.cat(figure_embeddings_list, dim=0)  # Concatenate
    except Exception as e:
        print(f"  Error during torch.cat: {e}")
        return -1.0

    print(f"  Concatenated embeddings shape: {figure_embeddings.shape}")

    # --- Add Assertion ---
    if figure_embeddings.shape[0] != len(eval_indices):
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"ERROR: Mismatch after encoding!")
         print(f"  Expected {len(eval_indices)} embeddings based on eval_indices.")
         print(f"  Got {figure_embeddings.shape[0]} embeddings after concatenation.")
         print(f"  This indicates some figures were skipped or failed during encoding.")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         return -1.0  # Return error code
    # --- End Assertion ---

    figure_embeddings = figure_embeddings.to(device)  # Move back to device

    ap_scores = []
    figures_evaluated = 0
    patents_found = 0
    # --- Calculate AP per query ---
    print(f"  Starting AP calculation for {len(eval_indices)} queries...")
    with torch.no_grad():
        for i in range(len(eval_indices)):  # Iterate from 0 to len(eval_indices)-1
            fig_idx = eval_indices[i]
            
            # Handle both single patent and multiple patents per figure
            true_patent_relative_idxs = []
            
            # Check if figure_to_pos_patent is a dict of lists (multiple positives per figure)
            if isinstance(figure_to_pos_patent.get(fig_idx, -1), list):
                true_patent_relative_idxs = figure_to_pos_patent.get(fig_idx, [])
            else:
                # Handle the case where it's a single value
                true_patent_idx = figure_to_pos_patent.get(fig_idx, -1)
                if true_patent_idx != -1:
                    true_patent_relative_idxs = [true_patent_idx]
            
            if not true_patent_relative_idxs:
                continue  # Skip if no positive patents
                
            # Filter patents that are within the valid range
            valid_patent_idxs = [idx for idx in true_patent_relative_idxs 
                                if patent_start_idx_rel <= idx < patent_end_idx_rel]
            
            if not valid_patent_idxs:
                continue  # Skip if no valid patents
                
            patents_found += 1

            # Access embedding using index i
            try:
                query_emb = figure_embeddings[i].unsqueeze(0)
            except IndexError as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"UNEXPECTED IndexError at query_emb = figure_embeddings[i]:")
                print(f"  i = {i}, len(eval_indices) = {len(eval_indices)}, figure_embeddings.shape = {figure_embeddings.shape}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue  # Skip this query

            # Calculate distances to all patent embeddings
            distances = pmath.dist(query_emb, all_patent_emb, k=model.k)
            distances = distances.squeeze(0)

            if torch.isnan(distances).any() or torch.isinf(distances).any():
                continue

            # Create target tensor with multiple positives
            target = torch.zeros(num_patents, device='cpu')
            for patent_idx in valid_patent_idxs:
                # Convert to local index within the patent slice
                target_local_idx = patent_idx - patent_start_idx_rel
                if 0 <= target_local_idx < num_patents:
                    target[target_local_idx] = 1
                else:
                    print(f"    Warning: target_local_idx {target_local_idx} out of bounds for target tensor size {num_patents}.")

            # Skip if no valid targets were set
            if torch.sum(target) == 0:
                continue

            scores = -distances.cpu()  # Convert distances to scores (smaller distance = higher score)

            # Calculate Average Precision
            target_np = target.numpy()
            scores_np = scores.numpy()
            try:
                ap = average_precision_score(target_np, scores_np)
                if not np.isnan(ap):
                    ap_scores.append(ap)
                    figures_evaluated += 1
            except ValueError as e:
                print(f"  ValueError during AP calculation for fig {fig_idx}: {e}")

    print(f"  Finished AP calculation. Calculated AP for {figures_evaluated} figures out of {patents_found} with valid patents found.")
    mean_ap = np.mean(ap_scores) if ap_scores else 0.0
    if not ap_scores and patents_found > 0:  # Add warning if patents were found but no AP calculated
        print("  WARNING: No valid AP scores were calculated despite finding valid patents.")
    return mean_ap


# Early stopping class implementation
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def compute_embeddings(model, X, implication, exclusion):
    """
    Extract embeddings from the trained hyperbolic model
    
    Args:
        model: Trained HMI model
        X: Node features tensor
        implication: Tensor of hierarchical relationships
        exclusion: Tensor of mutual exclusion relationships
        
    Returns:
        node_embeddings: Embeddings of all nodes in hyperbolic space
        label_embeddings: Embeddings of all labels in hyperbolic space
    """
    device = next(model.parameters()).device
    
    # Move data to device if needed
    if X.device != device:
        X = X.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get the encoded node embeddings
        encoded = model.ball.projx(X)
        node_embeddings = model.encoder(encoded)
        
        # Get the label embeddings directly from the model
        label_embeddings = model.label_emb
        
        # Compute the radii for visualization purposes
        node_dist = torch.norm(node_embeddings, p=2, dim=1)
        label_dist = torch.norm(label_embeddings, p=2, dim=1)
        
        node_radii = (1 - node_dist**2) / (2*node_dist)
        label_radii = (1 - label_dist**2) / (2*label_dist)
    
    # Convert to numpy for easier handling
    node_embeddings_np = node_embeddings.cpu().numpy()
    label_embeddings_np = label_embeddings.detach().numpy()
    node_radii_np = node_radii.detach().numpy()
    label_radii_np = label_radii.detach().numpy()
    
    return {
        'node_embeddings': node_embeddings_np,
        'label_embeddings': label_embeddings_np,
        'node_radii': node_radii_np,
        'label_radii': label_radii_np
    }

def calculate_hyperbolic_distances(model, X_figures_tensor, Y_pos, label_offsets, device, data_dir, num_samples=100): # Added data_dir
    """
    Calculate hyperbolic distances...
    """
    # ... (rest of the function setup) ...

    # Try to load implication data if available
    
    """
    Calculate hyperbolic distances between figures and their associated labels,
    as well as to random labels for comparison.
    
    Args:
        model: Trained hyperbolic embedding model
        X_figures_tensor: Tensor of figure features
        Y_pos: List of (figure_idx, patent_idx) pairs
        label_offsets: Dictionary mapping label types to their starting indices
        device: Device to run calculations on
        num_samples: Number of figure samples to analyze
    
    Returns:
        DataFrame with distance results
    """
    print(f"Calculating hyperbolic distances for {num_samples} sample figures...")
    model.eval()
    
    # Get label embeddings
    label_emb = model.label_emb.detach()
    
    # Calculate label type ranges
    patent_start_idx = 0  # Relative to label_emb
    num_patents = label_offsets.get('medium_cpcs', label_emb.shape[0]) - label_offsets.get('patents', 0)
    patent_end_idx = patent_start_idx + num_patents
    
    medium_start_idx = patent_end_idx
    num_medium_cpcs = label_offsets.get('big_cpcs', label_emb.shape[0]) - label_offsets.get('medium_cpcs', patent_end_idx)
    medium_end_idx = medium_start_idx + num_medium_cpcs
    
    big_start_idx = medium_end_idx
    num_big_cpcs = label_offsets.get('main_cpcs', label_emb.shape[0]) - label_offsets.get('big_cpcs', medium_end_idx)
    big_end_idx = big_start_idx + num_big_cpcs
    
    main_start_idx = big_end_idx
    num_main_cpcs = label_emb.shape[0] - main_start_idx
    
    print(f"Label counts: Patents={num_patents}, Medium CPCs={num_medium_cpcs}, Big CPCs={num_big_cpcs}, Main CPCs={num_main_cpcs}")
    
    # Create figure-to-patent mapping
    figure_to_patent = {}
    for fig_idx, patent_idx in Y_pos:
        figure_to_patent[fig_idx] = patent_idx
    

    # Create mapping dictionaries
    patent_to_medium = defaultdict(lambda: medium_start_idx)  # Default to first Medium CPC
    medium_to_big = defaultdict(lambda: big_start_idx)        # Default to first Big CPC
    big_to_main = defaultdict(lambda: main_start_idx)         # Default to first Main CPC
    
    # Try to load implication data if available
    try:
        npz_path = os.path.join(data_dir, 'training_data.npz')
        loaded_npz = np.load(npz_path)
        if 'implication' in loaded_npz:
            implications = loaded_npz['implication']
            print(f"Found {len(implications)} implication pairs")
            
            # Process implications to build hierarchy
            for child_idx, parent_idx in implications:
                if patent_start_idx <= child_idx < patent_end_idx and medium_start_idx <= parent_idx < medium_end_idx:
                    # Patent -> Medium CPC
                    patent_to_medium[child_idx] = parent_idx
                elif medium_start_idx <= child_idx < medium_end_idx and big_start_idx <= parent_idx < big_end_idx:
                    # Medium CPC -> Big CPC
                    medium_to_big[child_idx] = parent_idx
                elif big_start_idx <= child_idx < big_end_idx and main_start_idx <= parent_idx < main_start_idx + num_main_cpcs:
                    # Big CPC -> Main CPC
                    big_to_main[child_idx] = parent_idx
    except Exception as e:
        print(f"Could not load implication data: {e}")
        print("Using default hierarchical relationships")
    
    # Sample figures
    all_figures = list(figure_to_patent.keys())
    if len(all_figures) > num_samples:
        sampled_figures = random.sample(all_figures, num_samples)
    else:
        sampled_figures = all_figures
        print(f"Warning: Only {len(sampled_figures)} figures available with patent associations")
    
    # Prepare results storage
    results = []
    k_tensor = model.k.to(device)
    label_emb = label_emb.to(device) 
    # Calculate distances
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(sampled_figures), batch_size):
            batch_figures = sampled_figures[i:i+batch_size]
            
            # Get figure features and encode
            batch_X = X_figures_tensor[batch_figures].to(device)
            batch_embeddings = model.encode_figures(batch_X)
            
            for j, fig_idx in enumerate(batch_figures):
                fig_embedding = batch_embeddings[j:j+1]  # Keep batch dimension
                
                # Get associated labels
                patent_idx = figure_to_patent[fig_idx]
                medium_idx = patent_to_medium[patent_idx]
                big_idx = medium_to_big[medium_idx]
                main_idx = big_to_main[big_idx]
                
                # Get random labels for comparison
                random_patent_idx = random.randint(patent_start_idx, patent_end_idx-1)
                random_medium_idx = random.randint(medium_start_idx, medium_end_idx-1)
                random_big_idx = random.randint(big_start_idx, big_end_idx-1)
                random_main_idx = random.randint(main_start_idx, main_start_idx + num_main_cpcs - 1)
                while main_idx==random_main_idx:
                    random_main_idx = random.randint(main_start_idx, main_start_idx + num_main_cpcs - 1)
                # Calculate distances to true labels
                dist_to_patent = pmath.dist(fig_embedding, label_emb[patent_idx:patent_idx+1], k=k_tensor).item()
                dist_to_medium = pmath.dist(fig_embedding, label_emb[medium_idx:medium_idx+1], k=k_tensor).item()
                dist_to_big = pmath.dist(fig_embedding, label_emb[big_idx:big_idx+1], k=k_tensor).item()
                dist_to_main = pmath.dist(fig_embedding, label_emb[main_idx:main_idx+1], k=k_tensor).item()
                
                # Calculate distances to random labels
                dist_to_random_patent = pmath.dist(fig_embedding, label_emb[random_patent_idx:random_patent_idx+1], k=k_tensor).item()
                dist_to_random_medium = pmath.dist(fig_embedding, label_emb[random_medium_idx:random_medium_idx+1], k=k_tensor).item()
                dist_to_random_big = pmath.dist(fig_embedding, label_emb[random_big_idx:random_big_idx+1], k=k_tensor).item()
                dist_to_random_main = pmath.dist(fig_embedding, label_emb[random_main_idx:random_main_idx+1], k=k_tensor).item()
                
                # Store results
                results.append({
                    'figure_idx': fig_idx,
                    'patent_idx': patent_idx,
                    'medium_idx': medium_idx,
                    'big_idx': big_idx,
                    'main_idx': main_idx,
                    'dist_to_patent': dist_to_patent,
                    'dist_to_medium': dist_to_medium,
                    'dist_to_big': dist_to_big,
                    'dist_to_main': dist_to_main,
                    'dist_to_random_patent': dist_to_random_patent,
                    'dist_to_random_medium': dist_to_random_medium,
                    'dist_to_random_big': dist_to_random_big,
                    'dist_to_random_main': dist_to_random_main
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\nDistance Summary Statistics:")
    print(df[['dist_to_patent', 'dist_to_medium', 'dist_to_big', 'dist_to_main',
              'dist_to_random_patent', 'dist_to_random_medium', 'dist_to_random_big', 'dist_to_random_main']].describe())
    
    # Calculate average ratios (true/random)
    df['patent_ratio'] = df['dist_to_patent'] / df['dist_to_random_patent']
    df['medium_ratio'] = df['dist_to_medium'] / df['dist_to_random_medium']
    df['big_ratio'] = df['dist_to_big'] / df['dist_to_random_big']
    df['main_ratio'] = df['dist_to_main'] / df['dist_to_random_main']
    
    print("\nRatio Summary (True/Random):")
    print(df[['patent_ratio', 'medium_ratio', 'big_ratio', 'main_ratio']].describe())
    
    return df

def plot_distance_comparisons(df):
    """
    Create visualizations comparing true vs random distances.
    
    Args:
        df: DataFrame with distance results
    """
    # 1. Create paired boxplots for true vs random distances
    plt.figure(figsize=(14, 8))
    
    # Reshape data for seaborn
    plot_data = pd.DataFrame({
        'Patent (True)': df['dist_to_patent'],
        'Patent (Random)': df['dist_to_random_patent'],
        'Medium CPC (True)': df['dist_to_medium'],
        'Medium CPC (Random)': df['dist_to_random_medium'],
        'Big CPC (True)': df['dist_to_big'],
        'Big CPC (Random)': df['dist_to_random_big'],
        'Main CPC (True)': df['dist_to_main'],
        'Main CPC (Random)': df['dist_to_random_main']
    })
    
    # Melt for easier plotting
    plot_data_melted = pd.melt(plot_data)
    
    # Create boxplot
    sns.boxplot(x='variable', y='value', data=plot_data_melted)
    plt.xticks(rotation=45)
    plt.title('Hyperbolic Distances: True vs Random Labels')
    plt.xlabel('Label Type')
    plt.ylabel('Hyperbolic Distance')
    plt.tight_layout()
    plt.savefig('distance_comparison_boxplot.png', dpi=300)
    plt.show()
    
    # 2. Create violin plots for ratios
    plt.figure(figsize=(10, 6))
    ratio_data = pd.DataFrame({
        'Patent': df['patent_ratio'],
        'Medium CPC': df['medium_ratio'],
        'Big CPC': df['big_ratio'],
        'Main CPC': df['main_ratio']
    })
    ratio_data_melted = pd.melt(ratio_data)
    
    # Add reference line at ratio=1
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # Create violin plot
    sns.violinplot(x='variable', y='value', data=ratio_data_melted)
    plt.title('Ratio of True/Random Distances (Lower is Better)')
    plt.xlabel('Label Type')
    plt.ylabel('Distance Ratio (True/Random)')
    plt.tight_layout()
    plt.savefig('distance_ratio_violinplot.png', dpi=300)
    plt.show()
    
    # 3. Create distance progression plot
    plt.figure(figsize=(12, 6))
    
    # Calculate means for each distance type
    means = {
        'True': [
            df['dist_to_patent'].mean(),
            df['dist_to_medium'].mean(),
            df['dist_to_big'].mean(),
            df['dist_to_main'].mean()
        ],
        'Random': [
            df['dist_to_random_patent'].mean(),
            df['dist_to_random_medium'].mean(),
            df['dist_to_random_big'].mean(),
            df['dist_to_random_main'].mean()
        ]
    }
    
    # Plot means
    x = ['Patent', 'Medium CPC', 'Big CPC', 'Main CPC']
    plt.plot(x, means['True'], 'o-', label='True Association')
    plt.plot(x, means['Random'], 'o-', label='Random Association')
    plt.title('Average Hyperbolic Distance Progression')
    plt.xlabel('Label Type (Increasing Hierarchy Level)')
    plt.ylabel('Average Hyperbolic Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distance_progression.png', dpi=300)

def plot_embeddings_tsne(embeddings, metadata, title="t-SNE Visualization"):
    """
    Performs t-SNE and plots the 2D embeddings.

    Args:
        embeddings (np.ndarray): Combined embeddings (figures + labels).
        metadata (list or np.ndarray): List of labels for each embedding point.
        title (str): Title for the plot.
    """
    print(f"Running t-SNE on {embeddings.shape[0]} points...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    end_time = time.time()
    print(f"t-SNE finished in {end_time - start_time:.2f} seconds.")

    # Create plot
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(metadata)))
    colors = plt.cm.get_cmap('tab10', len(unique_labels)) # Use a colormap

    for i, label in enumerate(unique_labels):
        indices = [idx for idx, meta in enumerate(metadata) if meta == label]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    color=colors(i), label=label, alpha=0.6, s=10) # Smaller points

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig("tsne_visualization.png") # Optional: Save the plot
    plt.show()

def plot_embeddings_tsne_enhanced(embeddings, metadata, title="t-SNE Visualization", perplexity=30):
    """
    Performs t-SNE and plots the 2D embeddings with enhanced visuals.

    Args:
        embeddings (np.ndarray): Embeddings to plot.
        metadata (list or np.ndarray): List of labels for each embedding point.
        title (str): Title for the plot.
        perplexity (int): t-SNE perplexity parameter.
    """
    print(f"Running t-SNE on {embeddings.shape[0]} points (Perplexity={perplexity})...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, verbose=1, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)
    end_time = time.time()
    print(f"t-SNE finished in {end_time - start_time:.2f} seconds.")

    # Define distinct colors and markers
    label_styles = {
        'Figure': {'color': '#1f77b4', 'marker': '.', 'size': 5, 'alpha': 0.3}, # Muted blue, small dots
        'Patent': {'color': '#ff7f0e', 'marker': 'o', 'size': 15, 'alpha': 0.4}, # Orange circles
        'Medium CPC': {'color': '#2ca02c', 'marker': '^', 'size': 25, 'alpha': 0.7}, # Green triangles
        'Big CPC': {'color': '#d62728', 'marker': 's', 'size': 50, 'alpha': 0.8}, # Red squares
        'Main CPC': {'color': '#9467bd', 'marker': '*', 'size': 170, 'alpha': 1}, # Purple stars
    }
    default_style = {'color': 'gray', 'marker': 'x', 'size': 20, 'alpha': 0.5}

    # Create plot
    plt.figure(figsize=(14, 12))
    unique_labels = sorted(list(set(metadata)), key=lambda x: list(label_styles.keys()).index(x) if x in label_styles else 99)

    print("Plotting points...")
    for label in unique_labels:
        indices = [idx for idx, meta in enumerate(metadata) if meta == label]
        style = label_styles.get(label, default_style)
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    color=style['color'],
                    marker=style['marker'],
                    s=style['size'],
                    alpha=style['alpha'],
                    label=f"{label} ({len(indices)})") # Add count to label

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # Place legend outside the plot
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.savefig(f"{title.replace(' ', '_').lower()}_tsne.png", dpi=300) # Save the plot
    print(f"Plot saved as {title.replace(' ', '_').lower()}_tsne.png")
    plt.show()



def infer_model(model, X, A):
    """Test trained model.

    Load and test the model from the models folder. It calculates
    the ROC-Area Under the Curve (AUC) and the Average Precision (AP)

    Args:
        model
        X (torch.tensor)
        A (torch.tensor)

    Returns:
        Z (torch.tensor)
        A_reconstructed (torch.tensor)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    A = A.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        Z = model(X, A)

    return Z


# Main function to handle command-line arguments
def main():
    from auxiliary import loss_function_clamped_old, create_masks, hierarchical_triplet_loss, training_loss, neighborhood_contrastive_loss, enhanced_loss_function, evaluate_embeddings, extract_same_cpc_relationships, extract_parent_child_relationships,load_hyperbolic_inputs, mean_average_precision
    from models import VGAE,  EnhancedVGAE, HMI, EarlyStopping, HyperbolicEmbeddingModel, FigureOnlyHyperbolicModel, NPairBatchSampler, ImagePairDataset, collate_npairs, collate_enhanced_batch
    from process_graph import process_patent_graph, load_patent_graph

    import argparse
    import json
    import re
    import os
    import numpy as np
    import pandas as pd
    import torch
    import pickle
    import torch.nn as nn
    import geoopt as gt
    import random
    import time
    import geoopt.manifolds.stereographic.math as pmath
    from tqdm import tqdm
    from collections import defaultdict
    import torch.nn.functional as F
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.io as tvio
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPModel, CLIPProcessor
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import average_precision_score, roc_auc_score



    import torch as th
    import geoopt as gt
    from scipy.io import arff

    from matplotlib.patches import Circle
    import matplotlib.cm as cm
    import seaborn as sns

    import warnings; warnings.simplefilter('ignore')
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description="Train, load, or infer from a model.")
    parser.add_argument(
        "action", choices=["train", "train_gcn","train_hyp", "train_hyp_con", "train_end","train_end_2", "train_class", "plot","train_class_pro", "test", "infer", "dist"], help="Action to perform"
    )
    parser.add_argument("--model", choices=["GE","VGAE", "VGAE_W", "HMI"], help="Model to use")
    parser.add_argument("--path", type=str, help="Path to save/load the model")
    parser.add_argument(
        "--input_dim", type=int, default=512, help="Input size for the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden size for the model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Latent size for the model"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.005, help="Learning rate for the model"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Epochs to train the model"
    )

    args = parser.parse_args()
    

    # Select the model class based on user input
    

    if args.action == "train_class_pro":
        device='cuda'
        # Initialize the enhanced model
        model = EnhancedVGAE(args.input_dim, args.hidden_dim, args.latent_dim)
        model = model.to('cuda').float()     
        # Process the patent graph data
        (
            X,
            A_tilde_train,
        )=load_patent_graph("../data/2018/graph")
   
        X, A_tilde_train = load_patent_graph("../data/2018/graph")
        X              = X.float().to()
        A_tilde_train  = A_tilde_train.float().to(device)
        with open('figure_pair_connections.json', 'r') as f:
            pair_data = json.load(f)
        
        # Extract the sampled pairs
        sampled_pairs = pair_data['sampled_pairs']
        print(len(sampled_pairs))
        # Initialize the model
        
        # Train the model
        model = train_pair_classification_model(
            model, X, A_tilde_train, sampled_pairs, 
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            lr=args.learning_rate, batch_size=128
        ).to(device).float()
        
       
        # Save the trained model
        save_model(model, args.model, args.hidden_dim, args.latent_dim, args.learning_rate, args.epochs)
        parent_indices = extract_parent_child_relationships(A_tilde_train)
        print('Parent  indices computed')

        neighbor_indices = extract_same_cpc_relationships(A_tilde_train)

        print('Neighbor indices computed')
        # Save the trained model
        evaluate_embeddings(model, X, A_tilde_train, parent_indices, neighbor_indices)

    elif args.action == "train_hyp_con":
        import numpy as np 
        import geoopt.manifolds.stereographic.math as pmath
        with open('../notebooks/figure_to_pos_figures.pkl', 'rb') as f:
            figure_to_pos_figures = pickle.load(f)
        
        FEATURE_DIM = 512
        EMBED_DIM = 256
        LABEL_NUM = 14265
        all_anchor_indices = [idx for idx, pos_list in figure_to_pos_figures.items() if len(pos_list) > 0]
        random.shuffle(all_anchor_indices)
        val_count = int(0.1 * len(all_anchor_indices))  # 10% validation
        train_indices = all_anchor_indices[:-val_count]
        val_indices = all_anchor_indices[-val_count:]
        
        output_save_directory = '../notebooks/prepared_training_data'
      
        loaded_npz = np.load(os.path.join(output_save_directory, 'training_data.npz'))
        X_figures = loaded_npz['X_figures']

       
        model = FigureOnlyHyperbolicModel( 
            feature_num=FEATURE_DIM, #512 
            embed_dim=EMBED_DIM, #128 
            hidden_dims=[256, 128], 
            c=0.5, # Curvature 
            dropout_rate=0.05 
            )
       
       
        trained_model = train_hyperbolic_contrastive(
            model,
            X_figures,
            figure_to_pos_figures,
            train_indices,
            val_indices,
            epochs=12,
            batch_size=64,
            lr=5e-3,
            temperature=0.1,
            save_path="best_hyperbolic_model.pt",
            patience=3,
        )

    elif args.action == "train_hyp":
        import numpy as np
        import os
        # Initialize the enhanced model
        #model = HyperbolicPatentEmbedding(args.input_dim, args.hidden_dim, args.latent_dim)
        
        # Process the patent graph data
        #(
        #    X,
        #    A_tilde_train,
        #)=load_patent_graph("../data/2018/graph")
        # Extract hierarchical and neighborhood relationships
        # You'll need to implement these functions based on your graph structure
        # Load the figure pair connections
        # Train the model
    
        num_figures = 27101
        num_patents = 13552
        num_medium_cpc = 578
        num_big_cpc = 126
        num_main_cpc = 9
        
        

        output_save_directory = '../notebooks/prepared_training_data'
        try:
            loaded_npz = np.load(os.path.join(output_save_directory, 'training_data.npz'))
            X_figures_loaded = loaded_npz['X_figures']
            Y_pos_loaded = loaded_npz['Y_pos'].tolist() 
            Y_neg_loaded = loaded_npz['Y_neg'].tolist()
            implication_loaded = loaded_npz['implication'].tolist()
            exclusion_loaded = loaded_npz['exclusion'].tolist() 

            # --- ADD INDEX VALIDATION ---
            if implication_loaded:
                implication_np_arr = loaded_npz['implication'] # Use the numpy array directly
                min_idx = implication_np_arr.min()
                max_idx = implication_np_arr.max()
                print(f"Validation: Min index in implication_loaded: {min_idx}")
                print(f"Validation: Max index in implication_loaded: {max_idx}")
                # Check against expected label range (0 to LABEL_NUM - 1)
                LABEL_NUM = num_patents + num_medium_cpc + num_big_cpc + num_main_cpc
          # Make sure this is defined correctly based on your setup
                if min_idx < 0 or max_idx >= LABEL_NUM:
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"ERROR: Invalid indices found in implication data!")
                    print(f"  Expected range: [0, {LABEL_NUM - 1}]")
                    print(f"  Found range: [{min_idx}, {max_idx}]")
                    # Find specific problematic pairs
                    problematic_indices = implication_np_arr[(implication_np_arr < 0) | (implication_np_arr >= LABEL_NUM)]
                    print(f"  Problematic index values found: {np.unique(problematic_indices)}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # Decide whether to exit or filter
                    # exit() # Or filter the implication_loaded list
                else:
                    print("Validation: Implication indices seem within the expected range.")
            else:
                print("Validation: No implication pairs loaded.")

  
            # Access the new figure pair data
            if 'positive_figure_pairs' in loaded_npz:
                positive_figure_pairs=loaded_npz['positive_figure_pairs']
                print(f"Loaded {len(loaded_npz['positive_figure_pairs'])} positive figure pairs")
            if 'negative_figure_pairs' in loaded_npz:
                negative_figure_pairs=loaded_npz['negative_figure_pairs']
                print(f"Loaded {len(loaded_npz['negative_figure_pairs'])} negative figure pairs")


            with open(os.path.join(output_save_directory, 'label_offsets.json'), 'r') as f:
                label_offsets_loaded = json.load(f)
                print("Loaded data successfully.") 
        except:
            pass
        try:
            loaded_npz = np.load(os.path.join(output_save_directory, 'training_data.npz'))
            print("Loaded NPZ data keys:", list(loaded_npz.keys()))
            # Access data like: loaded_npz['X_figures'], loaded_npz['Y_pos']
            
            # Access the new figure pair data
            if 'positive_figure_pairs' in loaded_npz:
                print(f"Loaded {len(loaded_npz['positive_figure_pairs'])} positive figure pairs")
            if 'negative_figure_pairs' in loaded_npz:
                print(f"Loaded {len(loaded_npz['negative_figure_pairs'])} negative figure pairs")

            with open(os.path.join(output_save_directory, 'label_offsets.json'), 'r') as f:
                label_offsets_loaded = json.load(f)
            print("Loaded JSON offsets:", label_offsets_loaded)
        except FileNotFoundError:
            print("Saved files not found (this is expected if running the first time).")
        except Exception as e:
            print(f"Error loading saved files: {e}")

        # 2. Define Model Hyperparameters
        FEATURE_DIM = 512
        EMBED_DIM = 128 # Or 32, 128 etc.
        # Calculate total number of labels (patents + all CPC levels)
        num_patents = label_offsets_loaded['medium_cpcs'] - label_offsets_loaded['patents']
        num_medium_cpcs = label_offsets_loaded['big_cpcs'] - label_offsets_loaded['medium_cpcs']
        num_big_cpcs = label_offsets_loaded['main_cpcs'] - label_offsets_loaded['big_cpcs']
        # Need total count to find end of main_cpcs - assume it's the last type
        # This requires knowing the total number of nodes A was based on, or adjusting label_offsets
        # Let's assume main_cpcs count was stored or known:
        num_main_cpcs = 9 # From previous context
        LABEL_NUM = num_patents + num_medium_cpcs + num_big_cpcs + num_main_cpcs
        print(f"Feature Dim: {FEATURE_DIM}, Embed Dim: {EMBED_DIM}, Label Num: {LABEL_NUM}")


        # 3. Instantiate the Model
        model = HyperbolicEmbeddingModel(
            feature_num=FEATURE_DIM,
            embed_dim=EMBED_DIM,
            label_num=LABEL_NUM,
            c=2 # Curvature
        )

        

        # 4. Train the Model
        trained_model, final_loss = train_hyperbolic_retrieval_model(
            model=model,
            X_figures=X_figures_loaded,
            Y_pos=Y_pos_loaded,
            Y_neg=Y_neg_loaded,
            implication=implication_loaded,
            exclusion=exclusion_loaded,
            label_offsets=label_offsets_loaded,
            positive_figure_pairs=positive_figure_pairs,  # New: List of (figure_idx1, figure_idx2) from same patent
            negative_figure_pairs=negative_figure_pairs,
            epochs=150, 
            lr=6e-3, 
            batch_size=128, 
            num_neg_samples=1, 
            patience=10,
            figure_pair_weight=2,
            constraint_penalty=3,
            retrieval_penalty=2, 
            reg_penalty=0.01,
            validation_split=0.1,
            test_split=0.1,
            chunk_size=30000
        )
        print(f"Training finished. Final Test loss: {final_loss:.4f}")
    
    elif args.action == "train_end_2":
        import os
        import torch
        import json
        import torchvision.transforms as transforms

        from torch.utils.data import Dataset, DataLoader
        import torchvision
 
        from pathlib import Path
        from tqdm import tqdm
        # Configuration
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"  # or path to fine-tuned model
        FEATURE_DIM = 512  # CLIP feature dimension
        HYPERBOLIC_EMBED_DIM = 256  # Hyperbolic embedding dimension
        HYPERBOLIC_HIDDEN_DIMS = [256, 128]  # Hidden dimensions for hyperbolic encoder
        HYPERBOLIC_CURVATURE = 2  # Curvature of hyperbolic space
        
        BATCH_SIZE = 128  # Batch size (number of pairs * 2)
        EPOCHS = 10
        CLIP_LR = 2e-5
        HYPERBOLIC_LR = 5e-3
        TEMPERATURE = 0.07
        PATIENCE = 3
        
        CLIP_FINETUNE = True  # Whether to fine-tune CLIP
        CLIP_WEIGHT = 0.2  # Weight for CLIP loss
        HYPERBOLIC_WEIGHT = 1.0  # Weight for hyperbolic loss
        
        # Directories
        DATA_DIR = "../data/2018/"
        IMAGE_DIR = os.path.join(DATA_DIR, "test_query")
        SAVE_DIR = "models_storage/hyper/joint_clip_hyperbolic"
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load figure-to-positive-figures mapping
        # This should be a dictionary mapping figure names to lists of positive figure names
        # Example: {"figure1.jpg": ["figure2.jpg", "figure3.jpg"], ...}
        # You'll need to load this from your data
        output_save_directory = '../notebooks/prepared_training_data'
        with open(os.path.join(output_save_directory, 'label_offsets.json'), 'r') as f:
            label_offsets_loaded = json.load(f)
            print("Loaded JSON offsets:", label_offsets_loaded)

        num_patents = label_offsets_loaded['medium_cpcs'] - label_offsets_loaded['patents']
        num_medium_cpcs = label_offsets_loaded['big_cpcs'] - label_offsets_loaded['medium_cpcs']
        num_big_cpcs = label_offsets_loaded['main_cpcs'] - label_offsets_loaded['big_cpcs']
        # Need total count to find end of main_cpcs - assume it's the last type
        # This requires knowing the total number of nodes A was based on, or adjusting label_offsets
        # Let's assume main_cpcs count was stored or known:
        num_main_cpcs = 9 # From previous context
        LABEL_NUM = num_patents + num_medium_cpcs + num_big_cpcs + num_main_cpcs
        num_figures = 27101
        num_patents = 13552
        num_medium_cpc = 578
        num_big_cpc = 126
        num_main_cpc = 9
        
        

        output_save_directory = '../notebooks/prepared_training_data'
        try:
            loaded_npz = np.load(os.path.join(output_save_directory, 'training_data.npz'))
            X_figures_loaded = loaded_npz['X_figures']
            Y_pos_loaded = loaded_npz['Y_pos'].tolist() 
            Y_neg_loaded = loaded_npz['Y_neg'].tolist()
            implication_loaded = loaded_npz['implication'].tolist()
            exclusion_loaded = loaded_npz['exclusion'].tolist() 

            # --- ADD INDEX VALIDATION ---
            if implication_loaded:
                implication_np_arr = loaded_npz['implication'] # Use the numpy array directly
                min_idx = implication_np_arr.min()
                max_idx = implication_np_arr.max()
                print(f"Validation: Min index in implication_loaded: {min_idx}")
                print(f"Validation: Max index in implication_loaded: {max_idx}")
                # Check against expected label range (0 to LABEL_NUM - 1)
                LABEL_NUM = num_patents + num_medium_cpc + num_big_cpc + num_main_cpc
          # Make sure this is defined correctly based on your setup
                if min_idx < 0 or max_idx >= LABEL_NUM:
                    print(f"ERROR: Invalid indices found in implication data!")
                    print(f"  Expected range: [0, {LABEL_NUM - 1}]")
                    print(f"  Found range: [{min_idx}, {max_idx}]")
                    # Find specific problematic pairs
                    problematic_indices = implication_np_arr[(implication_np_arr < 0) | (implication_np_arr >= LABEL_NUM)]
                    print(f"  Problematic index values found: {np.unique(problematic_indices)}")
                else:
                    print("Validation: Implication indices seem within the expected range.")
            else:
                print("Validation: No implication pairs loaded.")

  
            # Access the new figure pair data
            if 'positive_figure_pairs' in loaded_npz:
                positive_figure_pairs=loaded_npz['positive_figure_pairs']
                print(f"Loaded {len(loaded_npz['positive_figure_pairs'])} positive figure pairs")
            if 'negative_figure_pairs' in loaded_npz:
                negative_figure_pairs=loaded_npz['negative_figure_pairs']
                print(f"Loaded {len(loaded_npz['negative_figure_pairs'])} negative figure pairs")


            with open(os.path.join(output_save_directory, 'label_offsets.json'), 'r') as f:
                label_offsets_loaded = json.load(f)
                print("Loaded data successfully.") 
        except:
            pass
        

        with open('../data/2018/ground_truth_2018_cpc.json', 'rb') as f:
            ground_truth = json.load(f)
        #print(figure_to_pos_figures)
        # Get image paths
        
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        all_query_image_paths = [str(path) for path in Path(IMAGE_DIR).rglob("*") if path.suffix in valid_extensions]
        print(len(all_query_image_paths))
        anchor_image_paths = []
        positive_image_paths = []

        for path in tqdm(all_query_image_paths):
            date_str = Path(path).name.split('-')[1]
            month = int(date_str[4:6])

            #if month <= 8 and path[24:] in image_dict:
            #if month <= 11 and path[24:] in image_dict:
            positives = ground_truth.get(Path(path).name, {}).get('patent_positives', [])
            if positives:
                anchor_image_paths.append(path)
                positive_image_paths.append(str(Path(IMAGE_DIR) / positives[0]))

        print(f"Total eligible pairs: {len(anchor_image_paths)}")
        combined = list(zip(anchor_image_paths, positive_image_paths))
        random.shuffle(combined)
        anchor_image_paths, positive_image_paths = zip(*combined)
        # ---- Patent-aware split logic ---- #
        def extract_patent_number_from_path(path):
            return Path(path).name.split('-')[0]

        def create_patent_aware_split(anchors, positives, val_ratio=0.2):
            patent_to_indices = {}
            for i, path in enumerate(anchors):
                patent = extract_patent_number_from_path(path)
                if patent not in patent_to_indices:
                    patent_to_indices[patent] = []
                patent_to_indices[patent].append(i)

            all_patents = list(patent_to_indices.keys())
            random.shuffle(all_patents)
            val_patents = set(all_patents[:int(len(all_patents) * val_ratio)])

            train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths = [], [], [], []

            for patent, indices in patent_to_indices.items():
                for i in indices:
                    if patent in val_patents:
                        val_anchor_paths.append(anchors[i])
                        val_positive_paths.append(positives[i])
                    else:
                        train_anchor_paths.append(anchors[i])
                        train_positive_paths.append(positives[i])

            return train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths

        def check_patent_overlap(train_paths, val_paths):
            train_patents = set(extract_patent_number_from_path(p) for p in train_paths)
            val_patents = set(extract_patent_number_from_path(p) for p in val_paths)
            overlap = train_patents.intersection(val_patents)
            return len(overlap), overlap

        train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths = create_patent_aware_split(
            anchor_image_paths, positive_image_paths, val_ratio=0.15
        )

        overlap_count, _ = check_patent_overlap(train_anchor_paths, val_anchor_paths)
        assert overlap_count == 0, "Patent overlap detected despite patent-aware split!"

        print(f"Training pairs: {len(train_anchor_paths)}")
        print(f"Validation pairs: {len(val_anchor_paths)}")

        # Define ImageDataset
        class ImageDataset(Dataset):
            def __init__(self, anchor_paths, positive_paths, transform=None):
                self.anchor_paths = anchor_paths
                self.positive_paths = positive_paths
                self.transform = transform

            def __len__(self):
                return len(self.anchor_paths)

            def __getitem__(self, idx):
                anchor_image = torchvision.io.read_image(self.anchor_paths[idx]).float() / 255.0
                positive_image = torchvision.io.read_image(self.positive_paths[idx]).float() / 255.0

                if anchor_image.shape[0] == 1:
                    anchor_image = anchor_image.repeat(3, 1, 1)
                if positive_image.shape[0] == 1:
                    positive_image = positive_image.repeat(3, 1, 1)

                if anchor_image.shape[0] == 4:
                    anchor_image = anchor_image[:3]
                if positive_image.shape[0] == 4:
                    positive_image = positive_image[:3]

                if self.transform:
                    anchor_image = self.transform(anchor_image)
                    positive_image = self.transform(positive_image)

                return idx, anchor_image, positive_image

      
        # Define transforms for tensor inputs
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Create datasets and dataloaders
        train_dataset = ImageDataset(train_anchor_paths, train_positive_paths, transform)
        val_dataset = ImageDataset(val_anchor_paths, val_positive_paths, transform_val)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False
        )
        
        # Load models
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        
        #hyperbolic_model = FigureOnlyHyperbolicModel(
        #    feature_num=FEATURE_DIM,
        #    embed_dim=HYPERBOLIC_EMBED_DIM,
        #    hidden_dims=HYPERBOLIC_HIDDEN_DIMS,
        #    c=HYPERBOLIC_CURVATURE,
        #    dropout_rate=DROPOUT_RATE
        #)

        # 3. Instantiate the Model
        hyperbolic_model = HyperbolicEmbeddingModel(
            feature_num=FEATURE_DIM,
            embed_dim=HYPERBOLIC_EMBED_DIM,
            label_num=LABEL_NUM,
            c=2 # Curvature
        )
        #hyperbolic_model_path='best_retrieval_model_c2_e256.pt'
        #hyperbolic_model.load_state_dict(torch.load(hyperbolic_model_path, map_location=device))
        
        # Train models end-to-end
        clip_model, hyperbolic_model = train_end_to_end_old(
            clip_model=clip_model,
            hyperbolic_model=hyperbolic_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            Y_pos=Y_pos_loaded,
            Y_neg=Y_neg_loaded,
            implication=implication_loaded,
            exclusion=exclusion_loaded,
            label_offsets=label_offsets_loaded,
            positive_figure_pairs=positive_figure_pairs,  # New: List of (figure_idx1, figure_idx2) from same patent
            negative_figure_pairs=negative_figure_pairs,
            
            epochs=EPOCHS,
            clip_lr=CLIP_LR,
            hyperbolic_lr=HYPERBOLIC_LR,
            temperature=TEMPERATURE,
            device=device,
            save_dir=SAVE_DIR,
            patience=PATIENCE,
            clip_finetune=CLIP_FINETUNE,
            clip_weight=CLIP_WEIGHT,
            hyperbolic_weight=HYPERBOLIC_WEIGHT,
            
            num_neg_samples=1, 
            figure_pair_weight=2,
            constraint_penalty=50,
            retrieval_penalty=50, 
            reg_penalty=0.1,
        )
        
        print("Training complete!")
        
    elif args.action == "train_end":
        import os
        import json
        import torch
        from torch.utils.data import Dataset, DataLoader
        import torchvision
        from torchvision import transforms
        from pathlib import Path
        from tqdm import tqdm

        # Configuration
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
        FEATURE_DIM = 512
        HYPERBOLIC_EMBED_DIM = 256
        HYPERBOLIC_HIDDEN_DIMS = [256, 128]
        HYPERBOLIC_CURVATURE = 2

        BATCH_SIZE = 128
        EPOCHS = 40
        CLIP_LR = 2e-5
        HYPERBOLIC_LR = 1e-3
        TEMPERATURE = 0.2
        PATIENCE = 5

        CLIP_FINETUNE = True
        CLIP_WEIGHT = 1.0
        HYPERBOLIC_WEIGHT = 0.0

        DATA_DIR = "../data/2018/"
        IMAGE_DIR = os.path.join(DATA_DIR, "test_query")
        SAVE_DIR = "models_storage/hyper/joint_clip_hyperbolic"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load CPC label offset metadata
        output_save_directory = '../notebooks/prepared_training_data'
        with open(os.path.join(output_save_directory, 'label_offsets.json'), 'r') as f:
            label_offsets_loaded = json.load(f)

        num_patents = label_offsets_loaded['medium_cpcs'] - label_offsets_loaded['patents']
        num_medium_cpcs = label_offsets_loaded['big_cpcs'] - label_offsets_loaded['medium_cpcs']
        num_big_cpcs = label_offsets_loaded['main_cpcs'] - label_offsets_loaded['big_cpcs']
        num_main_cpcs = 9  # Assumed known
        LABEL_NUM = num_patents + num_medium_cpcs + num_big_cpcs + num_main_cpcs

        # Load graph image embeddings
        #with open('../data/2018/graph_embeddings_clip_vit_base_patch16_patent_hierarchy.json', 'r') as f:
        #    image_dict = json.load(f)

        # Load ground truth
        with open('../data/2018/ground_truth_2018_cpc.json', 'r') as f:
            ground_truth = json.load(f)

        with open('graph_embeddings/image_ge_embeddings_GE_256_d512_l0.002_20_3_5ep.pkl', 'rb') as f:
            image_dict = pickle.load(f)
        # Prepare image pairs
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        all_query_image_paths = [str(path) for path in Path(IMAGE_DIR).rglob("*") if path.suffix in valid_extensions]
        print(len(all_query_image_paths))
        anchor_image_paths = []
        positive_image_paths = []

        for path in tqdm(all_query_image_paths):
            date_str = Path(path).name.split('-')[1]
            month = int(date_str[4:6])

            #if month <= 8 and path[24:] in image_dict:
            #if month <= 11 and path[24:] in image_dict:
            positives = ground_truth.get(Path(path).name, {}).get('patent_positives', [])
            if positives:
                anchor_image_paths.append(path)
                positive_image_paths.append(str(Path(IMAGE_DIR) / positives[0]))

        print(f"Total eligible pairs: {len(anchor_image_paths)}")
        combined = list(zip(anchor_image_paths, positive_image_paths))
        random.shuffle(combined)
        anchor_image_paths, positive_image_paths = zip(*combined)
        # ---- Patent-aware split logic ---- #
        def extract_patent_number_from_path(path):
            return Path(path).name.split('-')[0]

        def create_patent_aware_split(anchors, positives, val_ratio=0.2):
            patent_to_indices = {}
            for i, path in enumerate(anchors):
                patent = extract_patent_number_from_path(path)
                if patent not in patent_to_indices:
                    patent_to_indices[patent] = []
                patent_to_indices[patent].append(i)

            all_patents = list(patent_to_indices.keys())
            random.shuffle(all_patents)
            val_patents = set(all_patents[:int(len(all_patents) * val_ratio)])

            train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths = [], [], [], []

            for patent, indices in patent_to_indices.items():
                for i in indices:
                    if patent in val_patents:
                        val_anchor_paths.append(anchors[i])
                        val_positive_paths.append(positives[i])
                    else:
                        train_anchor_paths.append(anchors[i])
                        train_positive_paths.append(positives[i])

            return train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths

        def check_patent_overlap(train_paths, val_paths):
            train_patents = set(extract_patent_number_from_path(p) for p in train_paths)
            val_patents = set(extract_patent_number_from_path(p) for p in val_paths)
            overlap = train_patents.intersection(val_patents)
            return len(overlap), overlap

        train_anchor_paths, train_positive_paths, val_anchor_paths, val_positive_paths = create_patent_aware_split(
            anchor_image_paths, positive_image_paths, val_ratio=0.15
        )

        overlap_count, _ = check_patent_overlap(train_anchor_paths, val_anchor_paths)
        assert overlap_count == 0, "Patent overlap detected despite patent-aware split!"

        print(f"Training pairs: {len(train_anchor_paths)}")
        print(f"Validation pairs: {len(val_anchor_paths)}")

        # Define ImageDataset
        class ImageDataset(Dataset):
            def __init__(self, anchor_paths, positive_paths, transform=None):
                self.anchor_paths = anchor_paths
                self.positive_paths = positive_paths
                self.transform = transform

            def __len__(self):
                return len(self.anchor_paths)

            def __getitem__(self, idx):
                anchor_image = torchvision.io.read_image(self.anchor_paths[idx]).float() / 255.0
                positive_image = torchvision.io.read_image(self.positive_paths[idx]).float() / 255.0

                if anchor_image.shape[0] == 1:
                    anchor_image = anchor_image.repeat(3, 1, 1)
                if positive_image.shape[0] == 1:
                    positive_image = positive_image.repeat(3, 1, 1)

                if anchor_image.shape[0] == 4:
                    anchor_image = anchor_image[:3]
                if positive_image.shape[0] == 4:
                    positive_image = positive_image[:3]

                if self.transform:
                    anchor_image = self.transform(anchor_image)
                    positive_image = self.transform(positive_image)

                return anchor_image, positive_image

        
        # Define transforms for tensor inputs
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Create datasets and dataloaders
        train_dataset = ImageDataset(train_anchor_paths, train_positive_paths, transform)
        val_dataset = ImageDataset(val_anchor_paths, val_positive_paths, transform_val)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False
        )

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    
    
        hyperbolic_model = HyperbolicEmbeddingModel(
            feature_num=FEATURE_DIM,
            embed_dim=HYPERBOLIC_EMBED_DIM,
            label_num=LABEL_NUM,
            c=2 # Curvature
        )


        hyperbolic_model_path='best_retrieval_model_c2_e256.pt'
        hyperbolic_model.load_state_dict(torch.load(hyperbolic_model_path, map_location=device))
        
        # Train models end-to-end
        clip_model, hyperbolic_model = train_end_to_end(
            clip_model=clip_model,
            hyperbolic_model=hyperbolic_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=EPOCHS,
            clip_lr=CLIP_LR,
            hyperbolic_lr=HYPERBOLIC_LR,
            temperature=TEMPERATURE,
            device=device,
            save_dir=SAVE_DIR,
            patience=PATIENCE,
            clip_finetune=CLIP_FINETUNE,
            clip_weight=CLIP_WEIGHT,
            hyperbolic_weight=HYPERBOLIC_WEIGHT
        )
        
        print("Training complete!")
        

        
    
    elif args.action == "plot":
        import numpy as np
        import os

        #Configuration 
        data_dir = '../notebooks/prepared_training_data' 
        EMBED_DIM = 256
        CURVATURE_C = 2
        #model_path = f'best_retrieval_model_c{CURVATURE_C}_e{EMBED_DIM}.pt'
        model_path ="models_storage/hyper/joint_clip_hyperbolic/hyperbolic_model_e2e_full_256_2_.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PLOT_LABELS_ONLY = True

        
        TSNE_PERPLEXITY_LABELS = 20
        TSNE_PERPLEXITY_ALL = 50
        print(f"Using device: {device}")
        print(f"Plot labels only: {PLOT_LABELS_ONLY}")

    
        npz_path = os.path.join(data_dir, 'training_data.npz')
        json_path = os.path.join(data_dir, 'label_offsets.json')

        try:
            print(f"Loading data from {data_dir}...")
            loaded_npz = np.load(npz_path)
            X_figures_loaded = loaded_npz['X_figures']

            with open(json_path, 'r') as f:
                label_offsets_loaded = json.load(f)
            print("Data loaded successfully.")

        except FileNotFoundError:
            print(f"Error: Preprocessed data files not found in {data_dir}.")
            exit()
        except Exception as e:
            print(f"Error loading data: {e}")
            exit()

       
        FEATURE_DIM = X_figures_loaded.shape[1]
        
        num_patents = 0
        num_medium_cpcs = 0
        num_big_cpcs = 0
        num_main_cpcs = 0
        LABEL_NUM = 0

        if 'patents' in label_offsets_loaded:
            patent_start_abs = label_offsets_loaded['patents']
            # Find end of patents
            potential_next_offsets = [v for k, v in label_offsets_loaded.items() if v > patent_start_abs]
            patent_end_abs = min(potential_next_offsets) if potential_next_offsets else -1 # Need total node count if it's last

            if patent_end_abs != -1:
                num_patents = patent_end_abs - patent_start_abs
                if 'medium_cpcs' in label_offsets_loaded:
                    medium_start_abs = label_offsets_loaded['medium_cpcs']
                    potential_next_offsets = [v for k, v in label_offsets_loaded.items() if v > medium_start_abs]
                    medium_end_abs = min(potential_next_offsets) if potential_next_offsets else -1
                    if medium_end_abs != -1:
                        num_medium_cpcs = medium_end_abs - medium_start_abs
                        if 'big_cpcs' in label_offsets_loaded:
                            big_start_abs = label_offsets_loaded['big_cpcs']
                            potential_next_offsets = [v for k, v in label_offsets_loaded.items() if v > big_start_abs]
                            big_end_abs = min(potential_next_offsets) if potential_next_offsets else -1
                            if big_end_abs != -1:
                                num_big_cpcs = big_end_abs - big_start_abs
                                if 'main_cpcs' in label_offsets_loaded:
                                    main_start_abs = label_offsets_loaded['main_cpcs']
                                    num_main_cpcs = 9
        LABEL_NUM = num_patents + num_medium_cpcs + num_big_cpcs + num_main_cpcs
        if LABEL_NUM <= 0:
            # Fallback if calculation failed
            print("Warning: Could not reliably determine label counts from offsets. Using fallback values.")
            
            num_figures = 27101
            num_patents = 13552
            num_medium_cpc = 578
            num_big_cpc = 126
            num_main_cpc = 9
        print(f"Using Model Params: Feature Dim={FEATURE_DIM}, Embed Dim={EMBED_DIM}, Label Num={LABEL_NUM}, Curvature={CURVATURE_C}")
        print(f"Label Counts: Pat={num_patents}, Med={num_medium_cpcs}, Big={num_big_cpcs}, Main={num_main_cpcs}")

        #  Instantiate and Load Model 
        print("Instantiating model...")
        model = HyperbolicEmbeddingModel(
            feature_num=FEATURE_DIM,
            embed_dim=EMBED_DIM,
            label_num=LABEL_NUM,
            c=CURVATURE_C
        )
        model.to(device)

        try:
            print(f"Loading model state from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            exit()
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            print("Ensure the model architecture definition matches the saved checkpoint.")
            exit()

        #  Calculate Embeddings 
        print("Calculating embeddings...")
        with torch.no_grad():
            figure_embeddings_np = None
            if not PLOT_LABELS_ONLY:
                X_figures_tensor = torch.tensor(X_figures_loaded, dtype=torch.float32).to(device)
                # Add batching if necessary for large X_figures_tensor
                figure_embeddings_hyp = model.encode_figures(X_figures_tensor)
                figure_embeddings_np = figure_embeddings_hyp.cpu().numpy()
                print(f"  Calculated {figure_embeddings_np.shape[0]} figure embeddings.")
            else:
                print("  Skipping figure embedding calculation.")

            # Label Embeddings
            label_embeddings_hyp = model.label_emb
            label_embeddings_np = label_embeddings_hyp.detach().cpu().numpy()
            print(f"  Retrieved {label_embeddings_np.shape[0]} label embeddings.")
            if label_embeddings_np.shape[0] != LABEL_NUM:
                print(f"  Warning: Retrieved {label_embeddings_np.shape[0]} label embeddings but expected {LABEL_NUM}.")
                # Adjust counts if needed
                LABEL_NUM = label_embeddings_np.shape[0]

        # Analyze Hyperbolic Distances 
        print("Analyzing hyperbolic distances from origin (dist0)...")
        with torch.no_grad():
            # Calculate dist0 for all labels
            label_dist0 = model.ball.dist0(model.label_emb).cpu().numpy()
            
            # Calculate average dist0 for each label type
            patent_start_idx = 0
            patent_end_idx = num_patents
            medium_start_idx = patent_end_idx
            medium_end_idx = medium_start_idx + num_medium_cpcs
            big_start_idx = medium_end_idx
            big_end_idx = big_start_idx + num_big_cpcs
            main_start_idx = big_end_idx
            main_end_idx = main_start_idx + num_main_cpcs
            
            # Calculate average dist0 for each label type
            patent_dist0 = label_dist0[patent_start_idx:patent_end_idx]
            medium_dist0 = label_dist0[medium_start_idx:medium_end_idx]
            big_dist0 = label_dist0[big_start_idx:big_end_idx]
            main_dist0 = label_dist0[main_start_idx:main_end_idx]
            
            print(f"  Patent dist0: min={patent_dist0.min():.4f}, max={patent_dist0.max():.4f}, mean={patent_dist0.mean():.4f}")
            print(f"  Medium CPC dist0: min={medium_dist0.min():.4f}, max={medium_dist0.max():.4f}, mean={medium_dist0.mean():.4f}")
            print(f"  Big CPC dist0: min={big_dist0.min():.4f}, max={big_dist0.max():.4f}, mean={big_dist0.mean():.4f}")
            print(f"  Main CPC dist0: min={main_dist0.min():.4f}, max={main_dist0.max():.4f}, mean={main_dist0.mean():.4f}")
            
            # Plot dist0 distributions
            plt.figure(figsize=(12, 6))
            plt.hist(patent_dist0, bins=30, alpha=0.5, label=f'Patents (n={len(patent_dist0)})')
            plt.hist(medium_dist0, bins=30, alpha=0.5, label=f'Medium CPCs (n={len(medium_dist0)})')
            plt.hist(big_dist0, bins=30, alpha=0.5, label=f'Big CPCs (n={len(big_dist0)})')
            plt.hist(main_dist0, bins=30, alpha=0.5, label=f'Main CPCs (n={len(main_dist0)})')
            plt.xlabel('Distance from Origin (dist0)')
            plt.ylabel('Count')
            plt.title('Distribution of Hyperbolic Distances from Origin by Label Type')
            plt.legend()
            plt.tight_layout()
            plt.savefig('dist0_distributions.png', dpi=300)
            print("Saved dist0 distribution plot as dist0_distributions.png")

        # Combine Embeddings and Create Metadata
        print("Preparing data for t-SNE visualization...")
        if PLOT_LABELS_ONLY:
            # Only use label embeddings
            combined_embeddings = label_embeddings_np
            
            # Create metadata for labels
            metadata = []
            metadata.extend(['Patent'] * num_patents)
            metadata.extend(['Medium CPC'] * num_medium_cpcs)
            metadata.extend(['Big CPC'] * num_big_cpcs)
            metadata.extend(['Main CPC'] * num_main_cpcs)
            
            perplexity = TSNE_PERPLEXITY_LABELS
            title = "t-SNE of Label Embeddings (Hierarchy Visualization)"
        else:
            # Combine figure and label embeddings
            combined_embeddings = np.vstack((figure_embeddings_np, label_embeddings_np))
            
            # Create metadata for all points
            metadata = []
            metadata.extend(['Figure'] * figure_embeddings_np.shape[0])
            metadata.extend(['Patent'] * num_patents)
            metadata.extend(['Medium CPC'] * num_medium_cpcs)
            metadata.extend(['Big CPC'] * num_big_cpcs)
            metadata.extend(['Main CPC'] * num_main_cpcs)
            
            perplexity = TSNE_PERPLEXITY_ALL
            title = "t-SNE of All Embeddings (Figures & Labels)"

        # Verification
        if len(metadata) != combined_embeddings.shape[0]:
            print(f"Error: Metadata length ({len(metadata)}) does not match combined embeddings rows ({combined_embeddings.shape[0]})")
            print(f"  Num Figures: {0 if PLOT_LABELS_ONLY else figure_embeddings_np.shape[0]}")
            print(f"  Num Labels: {label_embeddings_np.shape[0]}")
            print(f"  Label Counts Used: Pat={num_patents}, Med={num_medium_cpcs}, Big={num_big_cpcs}, Main={num_main_cpcs}")
            exit()
        else:
            print(f"  Prepared metadata for {len(metadata)} points.")

        # Run t-SNE and Plot 
        plot_embeddings_tsne_enhanced(combined_embeddings, metadata, title=title, perplexity=perplexity)

        #  Plot Sub-Hierarchy Example (if not PLOT_LABELS_ONLY)
        if not PLOT_LABELS_ONLY and num_main_cpcs > 0 and num_big_cpcs > 0:
            print("\nCreating sub-hierarchy visualization...")
            try:
                # Select one Main CPC and its related hierarchy
                main_cpc_idx = main_start_idx  # First Main CPC
                
                
                related_big_cpcs = list(range(big_start_idx, min(big_start_idx + 5, big_end_idx)))
                
                # Find related Medium CPCs
                related_medium_cpcs = list(range(medium_start_idx, min(medium_start_idx + 10, medium_end_idx)))
                
                # Find related Patents (first few)
                related_patents = list(range(patent_start_idx, min(patent_start_idx + 20, patent_end_idx)))
                
                # Combine indices and get embeddings
                hierarchy_indices = [main_cpc_idx] + related_big_cpcs + related_medium_cpcs + related_patents
                hierarchy_embeddings = label_embeddings_np[hierarchy_indices]
                
                # Create metadata
                hierarchy_metadata = []
                hierarchy_metadata.extend(['Main CPC'] * 1)
                hierarchy_metadata.extend(['Big CPC'] * len(related_big_cpcs))
                hierarchy_metadata.extend(['Medium CPC'] * len(related_medium_cpcs))
                hierarchy_metadata.extend(['Patent'] * len(related_patents))
                
                # Plot
                plot_embeddings_tsne_enhanced(
                    hierarchy_embeddings, 
                    hierarchy_metadata, 
                    title="t-SNE of Sample Hierarchy", 
                    perplexity=min(5, len(hierarchy_indices)//2)  # Lower perplexity for fewer points
                )
            except Exception as e:
                print(f"Error creating sub-hierarchy visualization: {e}")

        print("Script finished.")

    elif args.action == "dist":
        import numpy as np
        import os
        # Configuration
        
        data_dir = '../notebooks/prepared_training_data' # Directory where data was saved
        EMBED_DIM = 256
        CURVATURE_C = 2
        #model_path = f'best_retrieval_model_c{CURVATURE_C}_e{EMBED_DIM}.pt'
        model_path ="models_storage/hyper/joint_clip_hyperbolic/hyperbolic_model_e2e_full_256_2_.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_SAMPLES = 4000  # Number of figures to analyze
        
        print(f"Using device: {device}")
        
        # Load data
        try:
            print(f"Loading data from {data_dir}...")
            loaded_npz = np.load(os.path.join(data_dir, 'training_data.npz'))
            X_figures_loaded = loaded_npz['X_figures']
            Y_pos_loaded = loaded_npz['Y_pos']
            
            with open(os.path.join(data_dir, 'label_offsets.json'), 'r') as f:
                label_offsets_loaded = json.load(f)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            exit()
        
        # Define model parameters
        FEATURE_DIM = X_figures_loaded.shape[1]
        
        
        
        # Calculate LABEL_NUM
        max_offset = max(label_offsets_loaded.values())
        # Assuming main_cpcs count is 9 as before
        num_figures = 27101
        num_patents = 13552
        num_medium_cpc = 578
        num_big_cpc = 126
        num_main_cpc = 9

        LABEL_NUM = num_patents + num_medium_cpc + num_big_cpc + num_main_cpc
        print(f"Using Model Params: Feature Dim={FEATURE_DIM}, Embed Dim={EMBED_DIM}, Label Num={LABEL_NUM}, Curvature={CURVATURE_C}")
        
        # Instantiate and load model
        print("Instantiating model...")
        model = HyperbolicEmbeddingModel(
            feature_num=FEATURE_DIM,
            embed_dim=EMBED_DIM,
            label_num=LABEL_NUM,
            c=CURVATURE_C
        )
        model.to(device)
        
        try:
            print(f"Loading model state from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
        
        # Convert features to tensor
        X_figures_tensor = torch.tensor(X_figures_loaded, dtype=torch.float32)
        
        # Calculate and analyze distances
        results_df = calculate_hyperbolic_distances(
        model=model,
        X_figures_tensor=X_figures_tensor,
        Y_pos=Y_pos_loaded,
        label_offsets=label_offsets_loaded,
        device=device,
        data_dir=data_dir, # Pass data_dir here
        num_samples=NUM_SAMPLES
        )
        
        # Save results to CSV
        results_df.to_csv('hyperbolic_distance_analysis.csv', index=False)
        print("Saved detailed results to hyperbolic_distance_analysis.csv")
        
        # Create visualizations
        plot_distance_comparisons(results_df)
        
        print("Analysis complete!")



    elif args.action == "test":
        if not args.path:
            print("Error: Path must be specified to load a model.")
            return
        model = load_model(args.path,args.latent_dim, args.hidden_dim)

        (
            X,
            A_tilde_train,
            _,
            test_edges,
            _,
            test_non_edges,
        ) = process_patent_graph()

        test_model(model, X, A_tilde_train, test_edges, test_non_edges)



    elif args.action == "infer":
        if not args.path:
            print("Error: Path must be specified to infer with a model.")
            return
        model = load_model(args.path, args.hidden_dim, args.latent_dim)
        infer_model(model)


if __name__ == "__main__":
    main()


#python train.py train_gcn --model GE --input_dim 512 --hidden_dim 256 --latent_dim 128 --learning_rate 0.01 --epochs 50
#python train.py train_gcn --model GE --input_dim 512 --hidden_dim 512 --latent_dim 512 --learning_rate 0.0001 --epochs 105
#python train.py train_class_pro --model GE --input_dim 512 --hidden_dim 256 --latent_dim 512 --learning_rate 0.002 --epochs 20
#python train.py train_hyp --model GE --input_dim 512 --hidden_dim 256 --latent_dim 512 --learning_rate 0.002 --epochs 2
#python compute_graph_embeddings.py