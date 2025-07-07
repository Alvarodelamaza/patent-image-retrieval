import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from collections import defaultdict
from sklearn.metrics import average_precision_score



def normalize_adjacency_dense_gpu(A):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = A.to(device)  # Move to GPU if available

    # Ensure self-loops
    A = A + torch.eye(A.size(0), device=A.device)

    # Convert to sparse format
    A = A.to_sparse()

    # Degree vector
    row_sum = torch.sparse.sum(A, dim=1).to_dense()

    # Avoid division by zero by adding a small epsilon
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(1e-10 + row_sum)).to(device)

    # Normalize adjacency
    normalized_A = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt

    # Enforce symmetry (optional but helps to handle numerical instability)
    normalized_A = (normalized_A + normalized_A.T) / 2.0

    return normalized_A

def loss_function_clamped_old(A, A_reconstructed, mu, log_sigma, beta=0.001):
    # Reconstruction loss (Binary Cross-Entropy)
    epsilon = 1e-7
    A_reconstructed = torch.clamp(A_reconstructed, min=epsilon, max=1 - epsilon)
    
    # Sum over all elements and normalize by number of elements
    num_elements = float(A.numel())
    recon_loss = F.binary_cross_entropy(A_reconstructed, A, reduction="sum") / num_elements
    
    # KL Divergence - normalize by number of nodes
    num_nodes = float(mu.size(0))
    kl_loss = -0.5 * torch.sum(
        1
        + log_sigma.clamp(min=-10, max=10)
        - mu.pow(2)
        - log_sigma.clamp(min=-10, max=10).exp()
    ) / num_nodes
    
    # Print components for debugging
    print(f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
    
    # Apply beta weighting to KL term
    return recon_loss + beta * kl_loss

def loss_function_with_annealing(A, A_reconstructed, mu, log_sigma, epoch, max_epochs=200, beta_min=0.0001, beta_max=0.001):
    # Calculate reconstruction loss
    epsilon = 1e-7
    A_reconstructed = torch.clamp(A_reconstructed, min=epsilon, max=1 - epsilon)
    num_elements = float(A.numel())
    recon_loss = F.binary_cross_entropy(A_reconstructed, A, reduction="sum") / num_elements
    
    # Calculate KL divergence
    num_nodes = float(mu.size(0))
    kl_loss = -0.5 * torch.sum(
        1 + log_sigma.clamp(min=-10, max=10) - mu.pow(2) - log_sigma.clamp(min=-10, max=10).exp()
    ) / num_nodes
    
    # Annealing factor that increases from beta_min to beta_max
    beta = beta_min + (beta_max - beta_min) * min(1.0, epoch / (max_epochs * 0.5))
    
    # Print components for debugging
    print(f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, Beta: {beta:.6f}")
    
    return recon_loss + beta * kl_loss

# New loss function that focuses on hierarchical and neighborhood relationships
def enhanced_loss_function(Z, parent_indices, neighbor_indices, epoch, max_epochs=200, beta_min=0.0001, beta_max=0.001):
    # Calculate KL divergence (regularization term from VAE)
    
    
    
    # Hierarchical loss: make embeddings close to their parent nodes
    hierarchical_loss = 0.0
    if parent_indices is not None:
        for node_idx, parent_idx in parent_indices:
            hierarchical_loss += torch.sum((Z[node_idx] - Z[parent_idx]) ** 2)
        hierarchical_loss /= len(parent_indices)
    
    # Neighborhood loss: make embeddings close to nodes that share the same CPC code
    neighborhood_loss = 0.0
    if neighbor_indices is not None:
        for node_idx, similar_idx in neighbor_indices:
            neighborhood_loss += torch.sum((Z[node_idx] - Z[similar_idx]) ** 2)
        neighborhood_loss /= len(neighbor_indices)
    
    # Annealing factor for KL divergence
    beta = beta_min + (beta_max - beta_min) * min(1.0, epoch / (max_epochs * 0.5))
    
    # Print components for debugging
    print(f"KL Loss: {kl_loss.item():.4f}, Hierarchical Loss: {hierarchical_loss:.4f}, Neighborhood Loss: {neighborhood_loss:.4f}, Beta: {beta:.6f}")
    

    alpha_h = 1.0  # Weight for hierarchical loss
    alpha_n = 1.0  # Weight for neighborhood loss
    
    return beta * kl_loss + alpha_h * hierarchical_loss + alpha_n * neighborhood_loss

def neighborhood_contrastive_loss(Z, neighbor_indices, temperature=0.07, eps=1e-8):
    """
    Numerically stable version of the contrastive loss
    """
    # Normalize embeddings for cosine similarity
    Z_norm = F.normalize(Z, p=2, dim=1)
    
    # Create similarity matrix with proper numerical stability
    sim_matrix = torch.matmul(Z_norm, Z_norm.t()) / temperature
    
    # Clip similarity values to prevent extreme values
    sim_matrix = torch.clamp(sim_matrix, min=-20.0, max=20.0)
    
    # Create positive mask from neighbor_indices
    batch_size = Z_norm.shape[0]
    pos_mask = torch.zeros((batch_size, batch_size), device=Z.device)
    
    for i, j in neighbor_indices:
        pos_mask[i.item(), j.item()] = 1
        pos_mask[j.item(), i.item()] = 1  # Symmetrical
    
    # Exclude self-connections
    self_mask = torch.eye(batch_size, device=Z.device)
    pos_mask = pos_mask * (1 - self_mask)
    
    # Compute loss with numerical stability
    exp_sim = torch.exp(sim_matrix)
    
    # Add small epsilon to prevent division by zero
    pos_sim = torch.sum(exp_sim * pos_mask, dim=1) + eps
    total_sim = torch.sum(exp_sim * (1 - self_mask), dim=1) + eps
    
    # Safe log computation
    log_prob = torch.log(pos_sim / total_sim)
    
    # Handle cases where a node has no positive neighbors
    has_pos = (torch.sum(pos_mask, dim=1) > 0).float()
    if torch.sum(has_pos) > 0:
        nce_loss = -torch.sum(log_prob * has_pos) / (torch.sum(has_pos) + eps)
    else:
        nce_loss = torch.tensor(0.0, device=Z.device)
    
    # Check for NaN and replace with zero if needed
    if torch.isnan(nce_loss):
        print("Warning: NaN detected in neighborhood loss, returning zero instead")
        nce_loss = torch.tensor(0.0, device=Z.device)
    
    return nce_loss

    
def hierarchical_triplet_loss(Z, parent_indices, margin=0.1):
 
    # Normalize embeddings
    Z_norm = F.normalize(Z, p=2, dim=1)
    
    # Extract child and parent embeddings
    child_embeddings = Z_norm[parent_indices[:, 0]]
    parent_embeddings = Z_norm[parent_indices[:, 1]]
    
    # Compute positive distances (child to parent)
    pos_distances = torch.sum((child_embeddings - parent_embeddings) ** 2, dim=1)
    
    # For each child, sample a random node that is not its parent as negative
    batch_size = parent_indices.shape[0]
    neg_distances = []
    
    for i in range(batch_size):
        child_idx = parent_indices[i, 0]
        parent_idx = parent_indices[i, 1]
        
        # Sample random node until we find one that's not the parent
        while True:
            neg_idx = torch.randint(0, Z.shape[0], (1,), device=Z.device)[0]
            if neg_idx != parent_idx:
                break
        
        # Compute distance to negative sample
        neg_distance = torch.sum((Z_norm[child_idx] - Z_norm[neg_idx]) ** 2)
        neg_distances.append(neg_distance)
    
    neg_distances = torch.stack(neg_distances)
    

    triplet_loss = torch.mean(torch.clamp(pos_distances - neg_distances + margin, min=0))
    
    return triplet_loss

def mean_average_precision(predictions, targets):
    """
    Calculate mean average precision for multi-label classification
    
    Args:
        predictions: Tensor of predicted probabilities (after sigmoid) [batch_size, num_labels]
        targets: Tensor of ground truth labels [batch_size, num_labels]
        
    Returns:
        Mean average precision score
    """
    # Move to CPU for calculation
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Calculate AP for each class and then take mean
    aps = []
    for i in range(targets.shape[1]):
        # Skip classes with no positive examples in the test set
        if np.sum(targets[:, i]) > 0:
            ap = average_precision_score(targets[:, i], predictions[:, i])
            aps.append(ap)
    
    # Return mean AP
    return np.mean(aps) if len(aps) > 0 else 0.0

def create_masks(num_nodes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    if seed is not None:
        np.random.seed(seed)  # For reproducibility

    # Random permutation of indices
    node_indices = np.random.permutation(num_nodes)

    # Split indices
    train_cutoff = int(train_ratio * num_nodes)
    val_cutoff = train_cutoff + int(val_ratio * num_nodes)

    train_indices = node_indices[:train_cutoff]
    val_indices = node_indices[train_cutoff:val_cutoff]
    test_indices = node_indices[val_cutoff:]

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def load_hyperbolic_inputs(filepath='hyperbolic_inputs.pkl'):
    """Load the hyperbolic model inputs from a pickle file."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        Y_pos = data['Y_pos']
        Y_neg = data['Y_neg']
        implication = data['implication']
        exclusion = data['exclusion']
        
        print(f"Loaded {len(Y_pos)} positive relationships")
        print(f"Loaded {len(Y_neg)} negative relationships")
        print(f"Loaded {len(implication)} implication relationships")
        print(f"Loaded {len(exclusion)} exclusion relationships")
        
        return Y_pos, Y_neg, implication, exclusion
    else:
        print(f"File {filepath} not found. Need to generate hyperbolic inputs first.")
        return None, None, None, None
def evaluate_embeddings(model, X, A_tilde, parent_indices, neighbor_indices):
    """
    Evaluate the quality of embeddings based on hierarchical and neighborhood preservation.
    
    Args:
        model: The trained EnhancedVGAE model
        X: Node features
        A_tilde: Adjacency matrix
        parent_indices: List/tensor of (child_idx, parent_idx) pairs
        neighbor_indices: List/tensor of (node_i, node_j) pairs that share the same CPC code
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device, dtype=torch.float64)
    A_tilde = A_tilde.to(device, dtype=torch.float64)
    
    # Get embeddings
    with torch.no_grad():
        Z = model(X, A_tilde)
    # Get embeddings
    with torch.no_grad():
         Z= model(X, A_tilde)
    
    # 1. Hierarchical Relationship Preservation
    if parent_indices is not None and len(parent_indices) > 0:
        # Convert to tensor if not already
        if not isinstance(parent_indices, torch.Tensor):
            parent_indices = torch.tensor(parent_indices, device=device)
            
        # Get child and parent embeddings
        child_embeddings = Z[parent_indices[:, 0]]
        parent_embeddings = Z[parent_indices[:, 1]]
        
        def cosine_similarity(x1, x2):
            # Normalize the vectors
            x1_normalized = x1 / torch.norm(x1, dim=1, keepdim=True)
            x2_normalized = x2 / torch.norm(x2, dim=1, keepdim=True)
            # Calculate cosine similarity
            return torch.sum(x1_normalized * x2_normalized, dim=1)

        # Calculate average cosine similarity between child and parent
        hier_similarities = cosine_similarity(child_embeddings, parent_embeddings)
        avg_hier_similarity = hier_similarities.mean().item()

        # Calculate random baseline (similarity between random pairs)
        num_samples = min(1000, len(parent_indices))
        random_indices = torch.randint(0, Z.size(0), (num_samples, 2), device=device)
        random_similarities = cosine_similarity(Z[random_indices[:, 0]], Z[random_indices[:, 1]])
        avg_random_similarity = random_similarities.mean().item()
        
        # Calculate hierarchical preservation ratio (should be < 1)
        hier_preservation_ratio = avg_hier_similarity / avg_random_similarity
        
        print(f"Hierarchical Preservation:")
        print(f"  Average Child-Parent Cosine-Sim: {avg_hier_similarity:.4f}")
        print(f"  Average Random Pair Cosine-Sim: {avg_random_similarity:.4f}")
        print(f"  Preservation Ratio: {hier_preservation_ratio:.4f} ")
    
    
    if neighbor_indices is not None and len(neighbor_indices) > 0:
        # Convert to tensor if not already
        if not isinstance(neighbor_indices, torch.Tensor):
            neighbor_indices = torch.tensor(neighbor_indices, device=device)
            
        # Get embeddings of nodes that should be neighbors
        node_embeddings = Z[neighbor_indices[:, 0]]
        neighbor_embeddings = Z[neighbor_indices[:, 1]]
        
        # Calculate average distance between nodes that should be neighbors
        neigh_distances = cosine_similarity(node_embeddings, neighbor_embeddings)
        avg_neigh_distance = neigh_distances.mean().item()
        
        # Calculate neighborhood preservation ratio
        neigh_preservation_ratio = avg_neigh_distance / avg_random_similarity
        
        print(f"Neighborhood Preservation:")
        print(f"  Average Same-CPC Cosine-Sim: {avg_neigh_distance:.4f}")
        print(f"  Average Random Pair Cosine-Sim: {avg_random_similarity:.4f}")
        print(f"  Preservation Ratio: {neigh_preservation_ratio:.4f} ")
    
   
    k_values = [1, 5, 10, 20]
    
    # Compute pairwise distances for all nodes
    pairwise_distances = torch.cdist(Z, Z, p=2)  # Euclidean distance
    
    # Set diagonal to infinity to exclude self
    pairwise_distances.fill_diagonal_(float('inf'))
    
    # For hierarchical relationships
    if parent_indices is not None and len(parent_indices) > 0:
        hier_hits_at_k = {k: 0 for k in k_values}
        
        for child_idx, parent_idx in parent_indices:
            # Get distances from this child to all nodes
            distances = pairwise_distances[child_idx]
            
            # Get indices of k nearest neighbors
            for k in k_values:
                _, topk_indices = torch.topk(distances, k, largest=False)
                if parent_idx in topk_indices:
                    hier_hits_at_k[k] += 1
        
        # Calculate hit ratio for each k
        for k in k_values:
            hit_ratio = hier_hits_at_k[k] / len(parent_indices)
            print(f"  Hierarchical Hit@{k}: {hit_ratio:.4f}")
    
    

def training_loss(Z, parent_indices, neighbor_indices, temp=0.1):
    Z_norm = F.normalize(Z, p=2, dim=1)
    
    # Hierarchical loss with temperature scaling
    hierarchical_loss = 0.0
    if parent_indices is not None and len(parent_indices) > 0:
        child_embeddings = Z_norm[parent_indices[:, 0]]
        parent_embeddings = Z_norm[parent_indices[:, 1]]
        
        # Positive pairs (child-parent)
        pos_sim = F.cosine_similarity(child_embeddings, parent_embeddings, dim=1)
        
        # Negative pairs (child with random nodes)
        batch_size = len(parent_indices)
        neg_indices = torch.randint(0, Z.size(0), (batch_size, 5), device=Z.device)  # 5 negative samples per positive
        neg_embeddings = Z_norm[neg_indices]
        neg_sim = torch.mean(torch.bmm(
            child_embeddings.unsqueeze(1),
            neg_embeddings.transpose(1, 2)
        ).squeeze(), dim=1)
        
        # InfoNCE loss
        hierarchical_loss = -torch.mean(
            pos_sim/temp - torch.log(torch.exp(pos_sim/temp) + torch.exp(neg_sim/temp))
        )

    # Neighborhood loss with temperature scaling
    neighborhood_loss = 0.0
    if neighbor_indices is not None and len(neighbor_indices) > 0:
        node_embeddings = Z_norm[neighbor_indices[:, 0]]
        neighbor_embeddings = Z_norm[neighbor_indices[:, 1]]
        
        # Positive pairs
        pos_sim = F.cosine_similarity(node_embeddings, neighbor_embeddings, dim=1)
        
        # Negative pairs
        batch_size = len(neighbor_indices)
        neg_indices = torch.randint(0, Z.size(0), (batch_size, 5), device=Z.device)
        neg_embeddings = Z_norm[neg_indices]
        neg_sim = torch.mean(torch.bmm(
            node_embeddings.unsqueeze(1),
            neg_embeddings.transpose(1, 2)
        ).squeeze(), dim=1)
        
        # InfoNCE loss
        neighborhood_loss = -torch.mean(
            pos_sim/temp - torch.log(torch.exp(pos_sim/temp) + torch.exp(neg_sim/temp))
        )
    
    return hierarchical_loss, neighborhood_loss

def extract_parent_child_relationships(A_tilde):

    num_figures = 22924
    num_patents = 11463
    num_medium_cpc = 566
    num_big_cpc = 126
    num_main_cpc = 9
    
    


    # Calculate boundaries for each type
    figures_start, figures_end = 0, num_figures
    patents_start, patents_end = num_figures, num_figures + num_patents
    medium_cpc_start, medium_cpc_end = patents_end, patents_end + num_medium_cpc
    big_cpc_start, big_cpc_end = medium_cpc_end, medium_cpc_end + num_big_cpc
    main_cpc_start, main_cpc_end = big_cpc_end, big_cpc_end + num_main_cpc

    # Get indices of nonzero edges in A_tilde
    I, J = torch.nonzero(A_tilde, as_tuple=True)

    # Create masks for each hierarchical relationship type
    mask_fig_pat = (I >= figures_start) & (I < figures_end) & \
                   (J >= patents_start) & (J < patents_end)
    
    mask_pat_med = (I >= patents_start) & (I < patents_end) & \
                   (J >= medium_cpc_start) & (J < medium_cpc_end)
    
    mask_med_big = (I >= medium_cpc_start) & (I < medium_cpc_end) & \
                   (J >= big_cpc_start) & (J < big_cpc_end)
    
    mask_big_main = (I >= big_cpc_start) & (I < big_cpc_end) & \
                    (J >= main_cpc_start) & (J < main_cpc_end)
    
    # Combine all masks
    combined_mask = mask_fig_pat | mask_pat_med | mask_med_big | mask_big_main
    
    # Extract the parent-child pairs
    parent_child_pairs = torch.stack([I[combined_mask], J[combined_mask]], dim=1)
    
    return parent_child_pairs

import torch
from collections import defaultdict

def extract_same_cpc_relationships(A_tilde):

    # Define node count and index ranges
    num_figures = 22924
    num_patents = 11463
    num_medium_cpc = 566
    
    # Calculate boundaries
    figures_start, figures_end = 0, num_figures
    patents_start, patents_end = num_figures, num_figures + num_patents
    medium_cpc_start, medium_cpc_end = patents_end, patents_end + num_medium_cpc
    
    # Get nonzero indices from adjacency matrix
    I, J = torch.nonzero(A_tilde, as_tuple=True)
    
    # Extract figure-patent connections
    mask_fig_pat = (I >= figures_start) & (I < figures_end) & (J >= patents_start) & (J < patents_end)
    fig_pat_edges = torch.stack([I[mask_fig_pat], J[mask_fig_pat]], dim=1)
    
    # Extract patent-medium CPC connections
    mask_pat_med = (I >= patents_start) & (I < patents_end) & (J >= medium_cpc_start) & (J < medium_cpc_end)
    pat_med_edges = torch.stack([I[mask_pat_med], J[mask_pat_med]], dim=1)
    
    # Create a mapping from patents to medium CPCs
    patent_to_medium_cpc = {}
    for patent, medium_cpc in pat_med_edges:
        patent_to_medium_cpc[patent.item()] = medium_cpc.item()
    
    # Create a mapping from medium CPCs to figures
    medium_cpc_to_figures = defaultdict(list)
    
    # For each figure-patent edge, find the medium CPC and add the figure to that CPC's list
    for figure, patent in fig_pat_edges:
        patent_item = patent.item()
        if patent_item in patent_to_medium_cpc:
            medium_cpc = patent_to_medium_cpc[patent_item]
            medium_cpc_to_figures[medium_cpc].append(figure.item())
    
    # Create pairs of figures that share the same medium CPC
    figure_pairs = []
    for medium_cpc, figures in medium_cpc_to_figures.items():
        if len(figures) > 1:  # Need at least 2 figures to make a pair
            for i in range(len(figures)):
                for j in range(i + 1, len(figures)):
                    figure_pairs.append((figures[i], figures[j]))
    
    # Convert to tensor if there are pairs
    if figure_pairs:
        return torch.tensor(figure_pairs)
    else:
        return torch.zeros((0, 2), dtype=torch.long)