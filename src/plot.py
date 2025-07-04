import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from models import VGAE, VGAE_W
from process_graph import process_patent_graph

def visualize_patent_embeddings(model_path, X, A_tilde, patent_indices, output_dir='tsne_embeddings', n_components=2, perplexity=30, random_state=42):
    """Load a trained VGAE model, encode the graph, and visualize specific patents using t-SNE."""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    print(f"📂 Loading model from {model_path}")
    model = VGAE(input_dim=517, hidden_dim=128, latent_dim=64)  # Adjust dimensions as needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    
    # Move data to the same device as the model
    X = X.to(device)
    A_tilde = A_tilde.to(device)
    
    # Define the output file path for t-SNE embeddings
    tsne_file_path = os.path.join(output_dir, f'embeddings_tsne{model_path[7:]}.npy')
    
    # Initialize embeddings variable
    embeddings = None
    
    # Check if t-SNE embeddings already exist
    if os.path.exists(tsne_file_path):
        print("🔄 Loading precomputed t-SNE embeddings...")
        embeddings_2d = np.load(tsne_file_path)
    else:
        # Get the latent representations
        print("🧠 Encoding graph to latent space...")
        with torch.no_grad():
            Z, _, mu, _ = model(X, A_tilde)
            embeddings = mu.cpu().numpy()
        
        # Apply t-SNE to reduce dimensionality for visualization
        print(f"🔄 Applying t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Save the t-SNE embeddings to a file
        os.makedirs(output_dir, exist_ok=True)
        np.save(tsne_file_path, embeddings_2d)
        print(f"✅ t-SNE embeddings saved to {tsne_file_path}")
    
    # Create a visualization
    plt.figure(figsize=(12, 10))
    
    # Plot all nodes as background
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.5, s=10)
    
    # Highlight the specified patents with different colors
    colors = sns.color_palette("husl", len(patent_indices))
    
    for i, idx in enumerate(patent_indices):
        if idx < len(embeddings_2d):  # Check if index is valid
            print(i,'ploted')
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                        c=[colors[i]], s=100, label=f"Patent {idx}", edgecolors='black')
        else:
            print(f"Warning: Index {idx} is out of bounds for embeddings_2d with shape {embeddings_2d.shape}.")
    
    plt.title("t-SNE Visualization of Patent Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Patent_embeddings_tsne_64.png", dpi=300)
    plt.show()
    
    print(f"✅ Visualization saved ")
    
    return embeddings, embeddings_2d

if __name__ == "__main__":
    try:
        (
            X,
            A_tilde_train,
            _,
            test_edges,
            _,
            test_non_edges,
        ) = process_patent_graph('../data/2018/graph')
        
        print(f"Loaded data: X shape {X.shape}, A_tilde_train shape {A_tilde_train.shape}")
        
        visualize_patent_embeddings(
            'models/VGAE_d64_0.001',
            X,
            A_tilde_train,
            [56297, 56796, 57096, 58796, 59096, 61296, 62296, 63296, 60896, 59196, 58796]
        )
    except Exception as e:
        print(f"An error occurred: {e}")