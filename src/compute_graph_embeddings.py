import torch
import numpy as np
import sys
import pickle
import os
from tqdm import tqdm
import torch.nn.functional as F

from auxiliary import loss_function_clamped_old, loss_function_with_annealing, enhanced_loss_function, evaluate_embeddings, extract_same_cpc_relationships, extract_parent_child_relationships
from models import VGAE,  EnhancedVGAE
from process_graph import process_patent_graph, load_patent_graph
from train import load_model, infer_model



def main():
    #model_name='GE_256_d512_l0.001_50'
    model_name='GE_256_d512_l0.002_20'
    # Create a dictionary to store the image embeddings
    image_embeddings = {}

    # Define the dictionary with images and their indices
    print('function initialized')
    # Load the dictionary
    with open('../notebooks/image_index_2018.pkl', 'rb') as file:
        image_dict = pickle.load(file)
    
    print('Index loaded')
    model = load_model(f'models/{model_name}',int(model_name[8:11]),int(model_name[3:6]))
    (
        X,
        A_tilde_train,
        
    ) = load_patent_graph("../data/2018/graph")
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    A_tilde_train = A_tilde_train.to(device)

    print('Generating embeddings...')
    with torch.no_grad():  # Add this for inference
        Z = infer_model(model, X, A_tilde_train)
        Z = F.normalize(Z, p=2, dim=1)

    # Create embeddings dictionary more efficiently
    print('Creating embeddings dictionary...')
    image_embeddings = {
        image_name: Z[index].cpu().numpy() 
        for image_name, index in tqdm(image_dict.items())
    }


    with open(f'graph_embeddings/image_ge_embeddings_{model_name}_3_5ep.pkl', 'wb') as f:
        pickle.dump(image_embeddings, f)
        print('Graph Latent Embeddings Saved')
    
    
    print(f'Graph Latent Embeddings Saved successfully')
    print(f'Number of embeddings saved: {len(image_embeddings)}')
    print(f'Embedding dimension: {next(iter(image_embeddings.values())).shape}')
    
    return image_embeddings

if __name__ == "__main__":
    main()