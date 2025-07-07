import os
import random
from itertools import combinations

import numpy as np
import torch
from scipy.sparse import coo_matrix, load_npz
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_dense_adj

from auxiliary import normalize_adjacency_dense_gpu

# We set the random seed to ensure reproducibility
torch.manual_seed(42)


def remove_edges_and_sample_optimized(
    edge_index, num_nodes, test_size=0.1, val_size=0.05, batch_size=10000
):
    # Convert edge_index to a list of edges
    edges = list(map(tuple, edge_index.t().tolist()))
    
    # Split edges into validation and test sets
    train_edges, temp_edges = train_test_split(
        edges, test_size=test_size + val_size, random_state=42
    )
    print("Edges train split")
    val_edges, test_edges = train_test_split(
        temp_edges, test_size=test_size / (test_size + val_size), random_state=42
    )
    print("Edges test/val split")
    
    # Convert edges to a set for faster lookup
    edges_set = set(edges)
    
    # Calculate how many non-edges we need
    num_val_non_edges = len(val_edges)
    num_test_non_edges = len(test_edges)
    total_non_edges_needed = num_val_non_edges + num_test_non_edges
    
    # Sample non-edges in batches
    val_non_edges = []
    test_non_edges = []
    non_edges_found = 0
    
    print(f"Sampling {total_non_edges_needed} non-edges in batches")
    
    # Process in batches of node pairs
    for i in range(0, num_nodes, batch_size):
        if non_edges_found >= total_non_edges_needed:
            break
            
        batch_end = min(i + batch_size, num_nodes)
        print(f"Processing nodes {i} to {batch_end-1}")
        
        for j in range(num_nodes):
            # Only process each pair once
            if j >= i and j < batch_end:
                continue
                
            # Generate potential non-edges from this batch to node j
            batch_pairs = [(min(n, j), max(n, j)) for n in range(i, batch_end)]
            
            # Filter out existing edges
            batch_non_edges = [pair for pair in batch_pairs if pair not in edges_set]
            
            # Add to our collections
            remaining_needed = total_non_edges_needed - non_edges_found
            batch_to_use = min(len(batch_non_edges), remaining_needed)
            
            if batch_to_use > 0:
                selected_non_edges = random.sample(batch_non_edges, batch_to_use)
                
                # Split between validation and test
                if len(val_non_edges) < num_val_non_edges:
                    val_to_add = min(batch_to_use, num_val_non_edges - len(val_non_edges))
                    val_non_edges.extend(selected_non_edges[:val_to_add])
                    test_non_edges.extend(selected_non_edges[val_to_add:])
                else:
                    test_non_edges.extend(selected_non_edges)
                
                non_edges_found += batch_to_use
                
                if non_edges_found >= total_non_edges_needed:
                    break
    
    print(f"Found {len(val_non_edges)} validation and {len(test_non_edges)} test non-edges")
    
    # Recreate training edge_index
    train_edge_index = torch.tensor(train_edges).t().contiguous()

    return (
        train_edge_index,
        val_edges,
        test_edges,
        val_non_edges,
        test_non_edges,
    )


def load_patent_graph(path):
    """Load PatentNet Dataset.

    Load the adjancecy matrix and feature matrix from a
    from a heterogeneous undirected patent dataset PatentNet

    Args:
        path (str)

    Returns:
        ``A``
        ``edge_index``
        ``X``
    """
    A = load_npz(f"{path}/combined_adj_query_hier_01_3.npz")
    X = load_npz(f"{path}/combined_features_matrix_query_hier_01_3_5ep.npz")
    X = torch.tensor(X.toarray(), dtype=torch.float32)
    A = torch.tensor(A.toarray(), dtype=torch.float32)
    A_coo = coo_matrix(A)
    edges = np.vstack((A_coo.row, A_coo.col)).T
    edge_index = torch.tensor(edges, dtype=torch.float32).T

    edge_index = edge_index.to(torch.int64)
    num_nodes = A.shape[0]
    train_adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

    A_tilde_train = normalize_adjacency_dense_gpu(
    train_adj_matrix.to(torch.float32))
    
    return X, A_tilde_train


def process_patent_graph(path="../data/2018/graph", processed_data_dir="../data/2018/graph/processed_graph_ge_1"):
    """Load Patnet Dataset.

    Load the adjancecy matrix and feature matrix from a
    from a heterogeneous undirected patent dataset PatentNet

    Args:
        path (str)

    Returns:
        ``A``
        ``edge_index``
        ``X``
    """
    # Create the directory for processed data if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # Define paths for processed files
    processed_X_path = os.path.join(processed_data_dir, "X.pt")
    processed_A_tilde_train_path = os.path.join(processed_data_dir, "A_tilde_train.pt")
    #processed_val_edges_path = os.path.join(processed_data_dir, "val_edges.npy")
    #processed_test_edges_path = os.path.join(processed_data_dir, "test_edges.npy")
    #processed_val_non_edges_path = os.path.join(processed_data_dir, "val_non_edges.npy")
    #processed_test_non_edges_path = os.path.join(
    #    processed_data_dir, "test_non_edges.npy"
    #)

   

    # Load the patent graph
    A, edge_index, X = load_patent_graph(path)
    print("Data Loaded")


    return (X, A_tilde_train) #val_edges, test_edges, val_non_edges, test_non_edges)
