import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import geoopt as gt
import wandb
import geoopt.manifolds.stereographic.math as pmath
import torchvision.transforms as transforms
import torchvision.io as tvio
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

torch.manual_seed(42)
# Constants
MIN_NORM = 1e-15
DROPOUT_RATE = 0.1 # Use a constant name

class ImagePairDataset(Dataset):
    """Dataset for loading image pairs with positive/negative relationships"""
    def __init__(self, image_paths, figure_to_pos_figures, transform=None):
        self.image_paths = image_paths
        self.figure_to_pos_figures = figure_to_pos_figures
        self.transform = transform
        
        # Create a mapping from image path to index
        self.path_to_idx = {path: idx for idx, path in enumerate(image_paths)}
        
        # Create pairs for training
        self.pairs = []
        
        for anchor_idx, anchor_path in enumerate(image_paths):
            anchor_name = os.path.basename(anchor_path)
            if anchor_name in list(figure_to_pos_figures.keys()):
                for pos_name in figure_to_pos_figures[anchor_name]['patent_positives']:
                    # Find the positive image path
                    pos_paths = [p for p in image_paths if os.path.basename(p) == pos_name]
                    if pos_paths:
                        pos_path = pos_paths[0]
                        pos_idx = self.path_to_idx[pos_path]
                        self.pairs.append((anchor_idx, pos_idx))
           
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx = self.pairs[idx]
        
        # Load anchor image
        anchor_path = self.image_paths[anchor_idx]
        try:
            anchor_img = self._load_and_transform_image(anchor_path)
        except Exception as e:
            print(f"Error loading anchor image {anchor_path}: {e}")
            # Return a placeholder or skip
            return None
        

        # Load positive image
        pos_path = self.image_paths[pos_idx]
        try:
            pos_img = self._load_and_transform_image(pos_path)
        except Exception as e:
            print(f"Error loading positive image {pos_path}: {e}")
            # Return a placeholder or skip
            return None
        
        return {
            'anchor_img': anchor_img,
            'pos_img': pos_img,
            'anchor_idx': anchor_idx,
            'pos_idx': pos_idx,
            'anchor_path': anchor_path,
            'pos_path': pos_path
        }
    
    def _load_and_transform_image(self, path):
        # Load image using torchvision.io
        image = tvio.read_image(path)
        
        # Convert to float and normalize to [0, 1]
        image = image.float() / 255.0
        
        # Handle grayscale images (1 channel)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # Handle RGBA images (4 channels)
        elif image.shape[0] == 4:
            image = image[:3]
            
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image

class NPairBatchSampler:
    """Sampler that creates N-pair batches for contrastive learning"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group pairs by anchor
        self.anchor_to_positives = defaultdict(list)
        for anchor_idx, pos_idx in dataset.pairs:
            self.anchor_to_positives[anchor_idx].append(pos_idx)
        
        self.anchors = list(self.anchor_to_positives.keys())
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.anchors)
        
        batch_anchors = []
        batch_positives = []
        
        for anchor_idx in self.anchors:
            if len(self.anchor_to_positives[anchor_idx]) > 0:
                # Select one positive randomly
                pos_idx = random.choice(self.anchor_to_positives[anchor_idx])
                
                batch_anchors.append(anchor_idx)
                batch_positives.append(pos_idx)
                
                if len(batch_anchors) == self.batch_size // 2:  # Each pair contributes 2 images
                    # Yield indices for the batch
                    batch = []
                    for a, p in zip(batch_anchors, batch_positives):
                        batch.extend([a, p])
                    yield batch
                    
                    # Reset batch
                    batch_anchors = []
                    batch_positives = []
        
        # Handle remaining items
        if batch_anchors:
            batch = []
            for a, p in zip(batch_anchors, batch_positives):
                batch.extend([a, p])
            yield batch
    
    def __len__(self):
        return (len(self.anchors) + self.batch_size // 2 - 1) // (self.batch_size // 2)

def collate_npairs(batch):
    """Custom collate function for N-pair batches"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Separate anchors and positives
    anchors = [item['anchor_img'] for item in batch]
    positives = [item['pos_img'] for item in batch]
    
    # Stack images
    anchor_imgs = torch.stack(anchors)
    positive_imgs = torch.stack(positives)
    
    # Combine for a single batch
    all_imgs = torch.cat([anchor_imgs, positive_imgs], dim=0)
    
    # Get indices
    anchor_indices = [item['anchor_idx'] for item in batch]
    pos_indices = [item['pos_idx'] for item in batch]
    
    # Get paths for reference
    anchor_paths = [item['anchor_path'] for item in batch]
    pos_paths = [item['pos_path'] for item in batch]
    
    return {
        'images': all_imgs,
        'n': len(anchors),  # Number of pairs
        'anchor_indices': anchor_indices,
        'pos_indices': pos_indices,
        'anchor_paths': anchor_paths,
        'pos_paths': pos_paths
    }

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # weight is *registered* as a parameter in float32
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A_tilde):
        # X, A_tilde must already share dtype & device
        X_trans = torch.matmul(X, self.weight)          # [N, out_features]
        return torch.matmul(A_tilde, X_trans)           # [N, out_features]


class InferenceModel(nn.Module):
    """
    Deep residual GCN encoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()

        # ---- input layer ----
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ---- hidden layers ----
        for _ in range(num_layers - 3):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ---- output layer ----
        self.layers.append(GCNLayer(hidden_dim, latent_dim))

    # ------------------------------------------------------------------
    def forward(self, X, A_tilde):
        """
        X           : [N, input_dim]  (float32)
        A_tilde     : [N, N]          (float32 row-normalised adj)
        """
        # ‼️ guarantee float32 everywhere
        X = X.float()
        A = A_tilde.float()

        # (optional) row normalisation on the fly
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

        # ---- first layer ----
        H = F.relu(self.bns[0](self.layers[0](X, A)))

        # ---- residual hidden layers ----
        for i in range(1, len(self.layers) - 1):
            H_new = F.relu(self.bns[i](self.layers[i](H, A)))
            H     = H + H_new

        # ---- output layer ----
        Z = self.layers[-1](H, A)
        return Z

# Set default dtype for numerical stability
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


# Constants
MIN_NORM = 1e-15
dropout = 0.1

# Mobius Linear Layer for hyperbolic space operations
class MobiusLinear(nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    k=-1.0,
):
    # Ensure consistent dtype
    weight = weight.to(input.dtype)
    if bias is not None:
        bias = bias.to(input.dtype)
    
    if hyperbolic_input:
        weight = F.dropout(weight, dropout)
        output = pmath.mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=k)
        output = pmath.mobius_add(output, bias, k=k)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = pmath.project(output, k=k)
    return output


# Early stopping utility
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path # Path to save the checkpoint

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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class HMI(nn.Module):
    
    def __init__(self, feature_num, hidden_size, embed_dim, label_num, **kwargs):
        super().__init__(**kwargs)
        
        self.ball = gt.PoincareBall(c=1.0)
        points = torch.randn(label_num, embed_dim) * 1e-5
        points = pmath.expmap0(points, k=self.ball.k)
        self.label_emb = gt.ManifoldParameter(points, manifold=self.ball)
        self.encoder = nn.Sequential(
            MobiusLinear(feature_num, embed_dim, bias=True, nonlin=None),
        )

    def regularization(self,points):
        return torch.norm(torch.norm(points, p=2, dim=1, keepdim=True) - 0.5, p=2, dim=1, keepdim=True)
    
    def radius_regularization(self,radius):
        return torch.norm(1-radius)

    def classifier(self,X):
        point_a = X.unsqueeze(1).expand(-1, self.label_emb.shape[0], -1) 
        point_b = self.label_emb.expand_as(point_a)
        logits = self.calculate_logit(point_a,point_b,dim=2).squeeze(2)
        return logits
    
    def forward(self, X,implication,exclusion):
        encoded = self.ball.projx(X)
        encoded = self.encoder(encoded)
        self.ball.assert_check_point_on_manifold(encoded)
        label_reg = self.regularization(self.label_emb)
        instance_reg = F.relu( torch.norm(encoded, p=2, dim=1, keepdim=True) - 0.99 ) + F.relu( 0.2 - torch.norm(encoded, p=2, dim=1, keepdim=True) )
        log_probability = self.classifier(encoded)
        # implication
        # Handle implication - check type first
        if isinstance(implication, tuple):
            implication = torch.tensor(implication, dtype=torch.long, device=X.device)
        
        if isinstance(exclusion, tuple):
            exclusion = torch.tensor(exclusion, dtype=torch.long, device=X.device)
        
        if implication.numel() > 0:  # Check if tensor is not empty
            if implication.dim() == 1:
                implication = implication.view(-1, 2)
                
            sub_label_id = implication[:,0]
            par_label_id = implication[:,1]
            sub_label_emb = self.label_emb[sub_label_id]
            par_label_emb = self.label_emb[par_label_id]
            inside_loss = F.relu(- self.insideness(sub_label_emb, par_label_emb))
        else:
            inside_loss = torch.tensor(0.0, device=X.device)
        
        if exclusion.numel() > 0:  # Check if tensor is not empty
            if exclusion.dim() == 1:
                exclusion = exclusion.view(-1, 2)
                
            left_label_id = exclusion[:,0]
            right_label_id = exclusion[:,1]
            left_label_emb = self.label_emb[left_label_id]
            right_label_emb = self.label_emb[right_label_id]
            disjoint_loss = F.relu(- self.disjointedness(left_label_emb, right_label_emb))
        else:
            disjoint_loss = torch.tensor(0.0, device=X.device)
        
        return log_probability, inside_loss.mean(), disjoint_loss.mean(), label_reg.mean(), instance_reg.mean()
    
    def insideness(self, point_a, point_b,dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_a = (1 - point_a_dist**2 )/ (2*point_a_dist )
        radius_b = (1 - point_b_dist**2 )/ (2*point_b_dist )
        center_a = point_a*(1 + radius_a/point_a_dist)
        center_b = point_b*(1 + radius_b/point_b_dist)
        center_dist = torch.norm(center_a-center_b,p=2,dim=dim,keepdim=True)
        insideness =  (radius_b - radius_a) - center_dist
        return insideness
    
    def disjointedness(self, point_a, point_b,dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_a = (1 - point_a_dist**2 )/ (2*point_a_dist )
        radius_b = (1 - point_b_dist**2 )/ (2*point_b_dist )
        center_a = point_a*(1 + radius_a/point_a_dist)
        center_b = point_b*(1 + radius_b/point_b_dist)
        center_dist = torch.norm(center_a-center_b,p=2,dim=dim,keepdim=True)
        disjointedness = center_dist - (radius_a + radius_b)
        return disjointedness
    
    def calculate_logit(self, points, label_emb,dim=-1):
        logit = self.insideness(points,label_emb) - self.disjointedness(points,label_emb)
        return logit

class DeeperHyperbolicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, c=1.0, dropout_rate=0.3):
        """
        A deeper hyperbolic encoder with multiple MobiusLinear layers.
        
        Args:
            input_dim: Dimension of input features (e.g., 512 from CLIP)
            hidden_dims: List of hidden dimensions (e.g., [256, 128])
            output_dim: Final embedding dimension
            c: Curvature parameter
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.c = c
        self.ball = gt.PoincareBall(c=c)
        self.k = torch.tensor([-c], dtype=torch.float32)
        self.dropout_rate = dropout_rate
        
        # First layer: Euclidean to hyperbolic
        self.first_layer = MobiusLinear(input_dim, hidden_dims[0], 
                                        hyperbolic_input=False, c=c)
        
        # Middle layers: hyperbolic to hyperbolic
        #self.middle_layers = nn.ModuleList()
        #for i in range(len(hidden_dims)-1):
        #    self.middle_layers.append(
        #        MobiusLinear(hidden_dims[i], hidden_dims[i+1], 
        #                    hyperbolic_input=True, c=c)
        #    )
        
        # Final layer: hyperbolic to hyperbolic (output)
        self.final_layer = MobiusLinear(hidden_dims[0], output_dim, 
                                       hyperbolic_input=True, c=c)
    
    def forward(self, x):
        # Move curvature tensor to the same device as input
        self.k = self.k.to(x.device)
        
        # Apply dropout to input
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # First layer: Euclidean to hyperbolic
        x = self.first_layer(x)
        # Apply hyperbolic activation
        x = pmath.mobius_fn_apply(torch.tanh, x, k=self.k)
        
        # Middle layers with activations
        #for layer in self.middle_layers:
        #    x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #    x = layer(x)
        #    x = pmath.mobius_fn_apply(torch.tanh, x, k=self.k)
        
        # Final layer
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_layer(x)
        
        # Ensure points are on the manifold
        x = pmath.project(x, k=self.k)
        return x

class HyperbolicEmbeddingModel(nn.Module):
    def __init__(self, feature_num, embed_dim, label_num,hidden_dims=[256, 128], c=1.0, **kwargs):
        """
        Args:
            feature_num: Dimensionality of input figure features (Euclidean).
            embed_dim: Dimensionality of the hyperbolic embedding space.
            label_num: Number of labels (patents + CPCs, etc.).
            c: Curvature of the Poincare ball.
        """
        super().__init__(**kwargs)
        self.c = c
        self.embed_dim = embed_dim
        self.k = torch.tensor([-c], dtype=torch.float32) # Store curvature tensor
        self.ball = gt.PoincareBall(c=self.c)

        # Embeddings for labels (patents, CPCs) - learnable parameters
        # Initialize them slightly away from origin for better separation initially
        label_points = torch.randn(label_num, embed_dim) * 0.1 # Increased initial spread
        label_points = pmath.expmap0(label_points, k=self.k)
        self.label_emb = gt.ManifoldParameter(label_points, manifold=self.ball)

        
        self.encoder = DeeperHyperbolicEncoder(
            input_dim=feature_num,
            hidden_dims=hidden_dims,
            output_dim=embed_dim,
            c=c,
            dropout_rate=DROPOUT_RATE
        )
        # Add more layers if needed, e.g., Mobius non-linearity + another MobiusLinear

    def encode_figures(self, features):
        """ Encodes Euclidean figure features into the hyperbolic space. """
        # Apply dropout to input features
        #print(f"Features: {features}, Type: {type(features)}")
        features = torch.tensor(features, dtype=torch.float32)
        features = F.dropout(features, p=DROPOUT_RATE, training=self.training)

        # Option 1: Simple Linear + ExpMap
        # tangent_vector = self.encoder(features)
        # encoded = pmath.expmap0(tangent_vector, k=self.k)
        # encoded = self.ball.projx(encoded) # Ensure projection

        # Option 2: Using MobiusLinear with hyperbolic_input=False
        encoded = self.encoder(features)
        # MobiusLinear already includes projection

        self.ball.assert_check_point_on_manifold(encoded)
        return encoded

    def calculate_hierarchical_loss(self, implication_pairs, exclusion_pairs):
        """ Calculates loss based on label hierarchy constraints. """
        inside_loss = torch.tensor(0.0, device=self.label_emb.device)
        disjoint_loss = torch.tensor(0.0, device=self.label_emb.device)
        num_labels = self.label_emb.shape[0] # Get the actual size

        # Implication Loss
        if implication_pairs is not None and implication_pairs.numel() > 0:
            if implication_pairs.dim() == 1: implication_pairs = implication_pairs.view(-1, 2)

            
            sub_label_idx = implication_pairs[:, 0]
            par_label_idx = implication_pairs[:, 1]
            min_sub, max_sub = sub_label_idx.min(), sub_label_idx.max()
            min_par, max_par = par_label_idx.min(), par_label_idx.max()

            if min_sub < 0 or max_sub >= num_labels or min_par < 0 or max_par >= num_labels:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR inside calculate_hierarchical_loss: Invalid indices detected!")
                print(f"  Label emb size: {num_labels}")
                print(f"  Sub indices range: [{min_sub.item()}, {max_sub.item()}]")
                print(f"  Par indices range: [{min_par.item()}, {max_par.item()}]")
                # Find specific bad indices
                bad_subs = sub_label_idx[(sub_label_idx < 0) | (sub_label_idx >= num_labels)]
                bad_pars = par_label_idx[(par_label_idx < 0) | (par_label_idx >= num_labels)]
                print(f"  Bad sub indices: {torch.unique(bad_subs)}")
                print(f"  Bad par indices: {torch.unique(bad_pars)}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # Raise error or handle appropriately
                raise IndexError("Invalid index detected in implication pairs")
           

            sub_label_emb = self.label_emb[sub_label_idx]
            par_label_emb = self.label_emb[par_label_idx]

            # Let's use a simpler distance constraint for now: d(sub, origin) < d(par, origin)
            # Or use the original HMI insideness if preferred, but ensure it's differentiable and stable
            # Using distance to origin as a proxy:
            dist_sub_origin = pmath.dist0(sub_label_emb, k=self.k)
            dist_par_origin = pmath.dist0(par_label_emb, k=self.k)
            # Margin loss: encourage dist_sub > dist_par (closer to boundary) - adjust if needed
            # inside_loss = F.relu(dist_par_origin - dist_sub_origin + 0.1).mean() # Example margin

            # Using original HMI insideness (ensure stability)
            insideness_val = self._hmi_insideness(sub_label_emb, par_label_emb)
            inside_loss = F.relu(-insideness_val + 0.05).mean() # Margin: insideness > 0.05

        # Exclusion Loss (Labels should be 'disjoint')
        if exclusion_pairs is not None and exclusion_pairs.numel() > 0:
            if exclusion_pairs.dim() == 1: exclusion_pairs = exclusion_pairs.view(-1, 2)
            left_label_idx = exclusion_pairs[:, 0]
            right_label_idx = exclusion_pairs[:, 1]
            left_label_emb = self.label_emb[left_label_idx]
            right_label_emb = self.label_emb[right_label_idx]
            # Use a margin-based loss: encourage disjointedness > margin
            # Using original HMI disjointedness (ensure stability)
            disjointedness_val = self._hmi_disjointedness(left_label_emb, right_label_emb)
            disjoint_loss = F.relu(-disjointedness_val + 0.1).mean() # Margin: disjointedness > 0.1

        return inside_loss, disjoint_loss

    def calculate_reg_loss(self, encoded_figures):
        """
        Calculates regularization losses using hyperbolic distance from the origin (dist0).
        """
        # Regularize label embeddings: prevent collapse to origin or explosion towards boundary
        # Use hyperbolic distance from origin (dist0)
        label_dist0 = self.ball.dist0(self.label_emb, dim=-1, keepdim=True).clamp_min(MIN_NORM)

        max_hyperbolic_dist_thresh = 8.0 # Penalize points too far from the origin
        min_hyperbolic_dist_thresh = 2 # Penalize collapsing to origin

        label_reg = (F.relu(min_hyperbolic_dist_thresh - label_dist0) + F.relu(label_dist0 - max_hyperbolic_dist_thresh)).mean()

        # Regularize figure embeddings: prevent explosion towards boundary
        figure_dist0 = self.ball.dist0(encoded_figures, dim=-1, keepdim=True).clamp_min(MIN_NORM)

        # Only penalize going too far from the origin (dist0 > max_thresh)
        instance_reg = F.relu(figure_dist0 - max_hyperbolic_dist_thresh).mean()

        return label_reg, instance_reg

    # Keep HMI geometry functions if using them for hierarchical loss
    def _hmi_insideness(self, point_a, point_b, dim=-1):
        # Ensure points are projected
        point_a = self.ball.projx(point_a)
        point_b = self.ball.projx(point_b)

        point_a_dist_origin = self.ball.dist0(point_a, dim=dim, keepdim=True).clamp_min(MIN_NORM)
        point_b_dist_origin = self.ball.dist0(point_b, dim=dim, keepdim=True).clamp_min(MIN_NORM)

        # Calculate Euclidean norm for radius/center calculations
       
        point_a_euclidean_norm = torch.norm(point_a, p=2, dim=dim, keepdim=True).clamp_min(MIN_NORM)
        point_b_euclidean_norm = torch.norm(point_b, p=2, dim=dim, keepdim=True).clamp_min(MIN_NORM)
        
        # Radii and centers calculation based on HMI paper (using Euclidean norms)
        k_tensor = self.k.to(point_a_euclidean_norm.device) # Use Euclidean norm's device
        sqrt_neg_k = torch.sqrt(-k_tensor)

        # Use Euclidean norms in the HMI formulas
        radius_a = (1 + k_tensor * point_a_euclidean_norm**2) / (2 * sqrt_neg_k * point_a_euclidean_norm)
        radius_b = (1 + k_tensor * point_b_euclidean_norm**2) / (2 * sqrt_neg_k * point_b_euclidean_norm)
        center_a = point_a * (1 + radius_a * sqrt_neg_k / point_a_euclidean_norm)
        center_b = point_b * (1 + radius_b * sqrt_neg_k / point_b_euclidean_norm)
        center_dist = torch.norm(center_a - center_b, p=2, dim=dim, keepdim=True) # Euclidean distance between centers

        insideness = (radius_b - radius_a) - center_dist
        return insideness

    def _hmi_disjointedness(self, point_a, point_b, dim=-1):
        point_a = self.ball.projx(point_a)
        point_b = self.ball.projx(point_b)

        point_a_euclidean_norm = torch.norm(point_a, p=2, dim=dim, keepdim=True).clamp_min(MIN_NORM)
        point_b_euclidean_norm = torch.norm(point_b, p=2, dim=dim, keepdim=True).clamp_min(MIN_NORM)
        

        k_tensor = self.k.to(point_a_euclidean_norm.device)
        sqrt_neg_k = torch.sqrt(-k_tensor)

        # Use Euclidean norms in the HMI formulas
        radius_a = (1 + k_tensor * point_a_euclidean_norm**2) / (2 * sqrt_neg_k * point_a_euclidean_norm)
        radius_b = (1 + k_tensor * point_b_euclidean_norm**2) / (2 * sqrt_neg_k * point_b_euclidean_norm)
        center_a = point_a * (1 + radius_a * sqrt_neg_k / point_a_euclidean_norm)
        center_b = point_b * (1 + radius_b * sqrt_neg_k / point_b_euclidean_norm)
        center_dist = torch.norm(center_a - center_b, p=2, dim=dim, keepdim=True) # Euclidean distance between centers

        disjointedness = center_dist - (radius_a + radius_b)
        return disjointedness

    def calculate_pair_loss(self, figure_embeddings, positive_pairs, negative_pairs):
        """
        Calculate cross-entropy loss for figure pairs.
        
        Args:
            figure_embeddings: Hyperbolic embeddings of figures [batch_size, embed_dim]
            positive_pairs: Tensor of indices for positive pairs [num_positives, 2]
            negative_pairs: Tensor of indices for negative pairs [num_negatives, 2]
            
        Returns:
            pair_loss: Cross-entropy loss for figure pairs
        """
        self.k = self.k.to(figure_embeddings.device)
        
        # If no pairs provided, return zero loss
        if positive_pairs is None or positive_pairs.numel() == 0:
            return torch.tensor(0.0, device=figure_embeddings.device)
            
        # Ensure pairs are properly shaped
        if positive_pairs.dim() == 1:
            positive_pairs = positive_pairs.view(-1, 2)
        if negative_pairs is not None and negative_pairs.numel() > 0:
            if negative_pairs.dim() == 1:
                negative_pairs = negative_pairs.view(-1, 2)
        
        # Combine positive and negative pairs
        all_pairs = positive_pairs
        if negative_pairs is not None and negative_pairs.numel() > 0:
            all_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
        
        # Create labels: 1 for positive pairs, 0 for negative pairs
        labels = torch.zeros(len(all_pairs), device=figure_embeddings.device)
        labels[:len(positive_pairs)] = 1.0
        
        # Calculate hyperbolic distances between pairs
        pair_distances = []
        for pair in all_pairs:
            idx1, idx2 = pair
            # Get embeddings for the pair
            emb1 = figure_embeddings[idx1].unsqueeze(0)  # Add batch dimension
            emb2 = figure_embeddings[idx2].unsqueeze(0)
            # Calculate hyperbolic distance
            dist = pmath.dist(emb1, emb2, k=self.k).squeeze()
            pair_distances.append(dist)
        
        # Stack distances into a tensor
        pair_distances = torch.stack(pair_distances)
        
        # Convert distances to similarities (smaller distance = higher similarity)
        similarities = -pair_distances / self.temperature
        
        # Group pairs by their first index (each query figure)
        unique_queries = torch.unique(all_pairs[:, 0])
        
        total_loss = 0.0
        num_queries = 0
        
        for query_idx in unique_queries:
            # Find all pairs with this query
            query_mask = (all_pairs[:, 0] == query_idx)
            if not query_mask.any():
                continue
                
            query_similarities = similarities[query_mask]
            query_labels = labels[query_mask]
            
            # Skip if no positive examples for this query
            if not query_labels.any():
                continue
                
            # Calculate cross-entropy loss
            # We want the positive pair to have higher similarity than negative pairs
            loss = F.cross_entropy(query_similarities.unsqueeze(0), 
                                  query_labels.argmax().unsqueeze(0))
            total_loss += loss
            num_queries += 1
        
        # Average loss over all queries
        if num_queries > 0:
            return total_loss / num_queries
        else:
            return torch.tensor(0.0, device=figure_embeddings.device)
    def forward(self, figure_features, implication_pairs=None, exclusion_pairs=None):
        """
        Main forward pass. Encodes figures and calculates hierarchical losses.

        Args:
            figure_features: Batch of Euclidean figure features.
            implication_pairs: Tensor of (child_idx, parent_idx) for labels.
            exclusion_pairs: Tensor of (label1_idx, label2_idx) for labels.

        Returns:
            encoded_figures: Hyperbolic embeddings for the input figures.
            inside_loss: Loss for implication constraints.
            disjoint_loss: Loss for exclusion constraints.
            label_reg: Regularization loss for label embeddings.
            instance_reg: Regularization loss for figure embeddings.
        """
        # Ensure curvature tensor is on the same device as input
        self.k = self.k.to(figure_features.device)

        encoded_figures = self.encode_figures(figure_features)
        #pair_loss = self.calculate_pair_loss(encoded_figures, positive_pairs, negative_pairs)
        
        # Calculate other losses
        #inside_loss, disjoint_loss = self.calculate_hierarchical_loss(implication_pairs, exclusion_pairs)
        #label_reg, instance_reg = self.calculate_reg_loss(encoded_figures)

        return encoded_figures#, pair_loss, inside_loss, disjoint_loss, label_reg, instance_reg



class FigureOnlyHyperbolicModel(nn.Module):
    def __init__(self, feature_num, embed_dim, hidden_dims=[256, 128], c=1.0, dropout_rate=0.3):
        super().__init__()
        self.c = c
        self.embed_dim = embed_dim
        self.k = torch.tensor([-c], dtype=torch.float32)
        self.ball = gt.PoincareBall(c=self.c)
        self.encoder = DeeperHyperbolicEncoder(
            input_dim=feature_num,
            hidden_dims=hidden_dims,
            output_dim=embed_dim,
            c=c,
            dropout_rate=dropout_rate
        )

    def encode_figures(self, features):
        features = F.dropout(features, p=self.encoder.dropout_rate, training=self.training)
        encoded = self.encoder(features)
        self.ball.assert_check_point_on_manifold(encoded)
        return encoded

    def calculate_pair_loss(self, figure_embeddings, positive_pairs, negative_pairs, temperature=0.07):
        self.k = self.k.to(figure_embeddings.device)
        if positive_pairs is None or positive_pairs.numel() == 0:
            return torch.tensor(0.0, device=figure_embeddings.device)
        if positive_pairs.dim() == 1:
            positive_pairs = positive_pairs.view(-1, 2)
        if negative_pairs is not None and negative_pairs.numel() > 0:
            if negative_pairs.dim() == 1:
                negative_pairs = negative_pairs.view(-1, 2)
        all_pairs = positive_pairs
        if negative_pairs is not None and negative_pairs.numel() > 0:
            all_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
        labels = torch.zeros(len(all_pairs), device=figure_embeddings.device)
        labels[:len(positive_pairs)] = 1.0
        pair_distances = []
        for pair in all_pairs:
            idx1, idx2 = pair
            emb1 = figure_embeddings[idx1].unsqueeze(0)
            emb2 = figure_embeddings[idx2].unsqueeze(0)
            dist = pmath.dist(emb1, emb2, k=self.k).squeeze()
            pair_distances.append(dist)
        pair_distances = torch.stack(pair_distances)
        similarities = -pair_distances / temperature
        return F.binary_cross_entropy_with_logits(similarities, labels.float())

    def forward(self, features):
        self.k = self.k.to(features.device)
        encoded_figures = self.encode_figures(features)
        # Return ONLY the encoded figures, not a tuple
        return encoded_figures
        
class EnhancedVGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        encoder = InferenceModel(input_dim, hidden_dim, latent_dim).to('cuda').float()
        # or, inside train_pair_classification_model, after constructing `model`:
        
        self.encoder = encoder
        # Simplified classification head for pair classification
        self.linear = nn.Linear(latent_dim * 2, latent_dim) 
        self.linear2 = nn.Linear(latent_dim, int(latent_dim / 2)) # *2 because we'll concatenate two embeddings
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        self.classifier = nn.Linear(int(latent_dim/2), 5)  # 5 classes for the 5 connection levels

    def forward(self, X, A_tilde):
        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        X = X.to(device)
        A_tilde = A_tilde.to(device)

        # Encode latent embeddings
        Z = self.encoder(X, A_tilde)

        # Normalize latent embeddings
        Z = F.normalize(Z, p=2, dim=1)
        
        return Z
    
    def classify_pair(self, z1, z2):
        # Concatenate the two embeddings
        pair_embedding = torch.cat([z1, z2], dim=1)
    
        # Classification layers with non-linearities and dropout
        hidden = F.relu(self.linear(pair_embedding))
        hidden = self.dropout(hidden)
        hidden2 = F.relu(self.linear2(hidden))
        hidden2 = self.dropout(hidden2)
        logits = self.classifier(hidden2)
        
        return logits

class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = InferenceModel(input_dim, hidden_dim, latent_dim)

    def forward(self, X, A_tilde):
        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        X = X.to(device)
        A_tilde = A_tilde.to(device)
        
        # Encode
        Z = self.encoder(X, A_tilde)
        
        
        
        # Normalize embeddings
        Z = F.normalize(Z, p=2, dim=1)

        # Reconstruct adjacency matrix
        A_reconstructed = torch.sigmoid(torch.matmul(Z, Z.T))
        
        return Z, A_reconstructed
class EnhancedHyperbolicDataset(torch.utils.data.Dataset):
    """
    Dataset for training with multiple hyperbolic losses.
    Handles image pairs, patent labels, and figure pairs.
    """
    def __init__(
        self, 
        image_paths, 
        figure_to_pos_figures, 
        figure_to_patent,  # Mapping from figure to patent ID
        patent_to_label,   # Mapping from patent ID to label indices
        label_hierarchy,   # Dictionary representing label hierarchy
        transform=None
    ):
        self.image_paths = image_paths
        self.figure_to_pos_figures = figure_to_pos_figures
        self.figure_to_patent = figure_to_patent
        self.patent_to_label = patent_to_label
        self.label_hierarchy = label_hierarchy
        self.transform = transform
        
        # Create a mapping from image path to index
        self.path_to_idx = {path: idx for idx, path in enumerate(image_paths)}
        
        # Create pairs for contrastive learning
        self.pairs = []
        
        print(f"Creating pairs from {len(image_paths)} images and {len(figure_to_pos_figures)} figure mappings")
        
        for anchor_idx, anchor_path in enumerate(image_paths):
            anchor_name = os.path.basename(anchor_path)
            if anchor_name in figure_to_pos_figures:
                for pos_name in figure_to_pos_figures[anchor_name]:
                    # Find the positive image path
                    pos_paths = [p for p in image_paths if os.path.basename(p) == pos_name]
                    if pos_paths:
                        pos_path = pos_paths[0]
                        pos_idx = self.path_to_idx[pos_path]
                        self.pairs.append((anchor_idx, pos_idx))
        
        print(f"Created {len(self.pairs)} pairs")
        
        # Create figure-to-figure pairs (positive and negative)
        self.pos_fig_pairs = []
        self.neg_fig_pairs = []
        
        # Create positive figure pairs (same patent)
        for patent_id, figures in self.group_figures_by_patent().items():
            if len(figures) >= 2:
                # Create positive pairs from figures in the same patent
                for i in range(len(figures)):
                    for j in range(i+1, len(figures)):
                        if figures[i] in self.path_to_idx and figures[j] in self.path_to_idx:
                            self.pos_fig_pairs.append((self.path_to_idx[figures[i]], self.path_to_idx[figures[j]]))
        
        # Create negative figure pairs (different patents, different labels)
        patent_groups = list(self.group_figures_by_patent().items())
        if len(patent_groups) >= 2:
            # Sample some negative pairs (not exhaustive to avoid too many pairs)
            num_neg_pairs = min(len(self.pos_fig_pairs) * 3, 10000)  # Cap at reasonable number
            
            for _ in range(num_neg_pairs):
                # Pick two different patents
                idx1, idx2 = random.sample(range(len(patent_groups)), 2)
                patent1, figures1 = patent_groups[idx1]
                patent2, figures2 = patent_groups[idx2]
                
                # Check if patents have different labels
                label1 = self.patent_to_label.get(patent1, None)
                label2 = self.patent_to_label.get(patent2, None)
                
                if label1 is not None and label2 is not None and label1 != label2:
                    # Pick a random figure from each patent
                    if figures1 and figures2:
                        fig1 = random.choice(figures1)
                        fig2 = random.choice(figures2)
                        
                        if fig1 in self.path_to_idx and fig2 in self.path_to_idx:
                            self.neg_fig_pairs.append((self.path_to_idx[fig1], self.path_to_idx[fig2]))
        
        print(f"Created {len(self.pos_fig_pairs)} positive figure pairs")
        print(f"Created {len(self.neg_fig_pairs)} negative figure pairs")
        
        # Create implication tensor for label hierarchy
        self.implication_tensor = self.create_implication_tensor()
        
        # Create label-to-index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.patent_to_label.values())))}
        
        if len(self.pairs) == 0:
            print("WARNING: No valid pairs found! Dataset will not work.")
    
    def group_figures_by_patent(self):
        """Group figures by their patent ID"""
        patent_to_figures = {}
        for path in self.image_paths:
            figure_name = os.path.basename(path)
            patent_id = self.figure_to_patent.get(figure_name)
            if patent_id:
                if patent_id not in patent_to_figures:
                    patent_to_figures[patent_id] = []
                patent_to_figures[patent_id].append(path)
        return patent_to_figures
    
    def create_implication_tensor(self):
        """Create implication tensor from label hierarchy"""
        labels = sorted(set(self.patent_to_label.values()))
        num_labels = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Initialize implication tensor
        implication = torch.zeros((num_labels, num_labels))
        
        # Fill diagonal (each label implies itself)
        for i in range(num_labels):
            implication[i, i] = 1
        
        # Fill hierarchy implications
        for child, parents in self.label_hierarchy.items():
            if child in label_to_idx:
                child_idx = label_to_idx[child]
                for parent in parents:
                    if parent in label_to_idx:
                        parent_idx = label_to_idx[parent]
                        implication[child_idx, parent_idx] = 1
        
        return implication
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx = self.pairs[idx]
        
        anchor_path = self.image_paths[anchor_idx]
        pos_path = self.image_paths[pos_idx]
        
        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
        
        # Get patent labels
        anchor_name = os.path.basename(anchor_path)
        pos_name = os.path.basename(pos_path)
        
        anchor_patent = self.figure_to_patent.get(anchor_name)
        pos_patent = self.figure_to_patent.get(pos_name)
        
        # Get label indices
        anchor_label = None
        pos_label = None
        
        if anchor_patent and anchor_patent in self.patent_to_label:
            anchor_label = self.patent_to_label[anchor_patent]
            if anchor_label in self.label_to_idx:
                anchor_label = self.label_to_idx[anchor_label]
        
        if pos_patent and pos_patent in self.patent_to_label:
            pos_label = self.patent_to_label[pos_patent]
            if pos_label in self.label_to_idx:
                pos_label = self.label_to_idx[pos_label]
        
        # Sample negative patent (different from anchor and positive)
        neg_label = None
        while neg_label is None or neg_label == anchor_label or neg_label == pos_label:
            # Sample a random label
            neg_label = random.choice(list(self.label_to_idx.values()))
        
        # Sample figure pairs for this batch
        batch_pos_fig_pairs = []
        batch_neg_fig_pairs = []
        
        # Sample a subset of positive figure pairs
        if self.pos_fig_pairs:
            num_pos_pairs = min(len(self.pos_fig_pairs), 32)  # Limit number of pairs
            batch_pos_fig_pairs = random.sample(self.pos_fig_pairs, num_pos_pairs)
        
        # Sample a subset of negative figure pairs
        if self.neg_fig_pairs:
            num_neg_pairs = min(len(self.neg_fig_pairs), 32)  # Limit number of pairs
            batch_neg_fig_pairs = random.sample(self.neg_fig_pairs, num_neg_pairs)
        
        # Convert to tensors
        batch_pos_fig_pairs = torch.tensor(batch_pos_fig_pairs) if batch_pos_fig_pairs else None
        batch_neg_fig_pairs = torch.tensor(batch_neg_fig_pairs) if batch_neg_fig_pairs else None
        
        return {
            'images': torch.stack([anchor_img, pos_img]),
            'n': 1,  # Number of pairs (always 1 in this case)
            'pos_patents': torch.tensor([anchor_label]) if anchor_label is not None else None,
            'neg_patents': torch.tensor([neg_label]) if neg_label is not None else None,
            'pos_fig_pairs': batch_pos_fig_pairs,
            'neg_fig_pairs': batch_neg_fig_pairs
        }

def collate_enhanced_batch(batch):
    """
    Custom collate function for the enhanced dataset
    """
    if len(batch) == 0:
        return None
    
    # Stack all images
    all_images = torch.cat([item['images'] for item in batch])
    
    # Number of pairs
    n = len(batch)
    
    # Collect patent labels
    pos_patents = [item['pos_patents'] for item in batch if item['pos_patents'] is not None]
    neg_patents = [item['neg_patents'] for item in batch if item['neg_patents'] is not None]
    
    pos_patents = torch.cat(pos_patents) if pos_patents else None
    neg_patents = torch.cat(neg_patents) if neg_patents else None
    
    # Collect figure pairs
    pos_fig_pairs = [item['pos_fig_pairs'] for item in batch if item['pos_fig_pairs'] is not None]
    neg_fig_pairs = [item['neg_fig_pairs'] for item in batch if item['neg_fig_pairs'] is not None]
    
    # Adjust indices for the batch
    if pos_fig_pairs:
        # Adjust indices to account for batch concatenation
        for i, pairs in enumerate(pos_fig_pairs):
            pos_fig_pairs[i] = pairs + 2 * i  # Each item adds 2 images (anchor + positive)
    
    if neg_fig_pairs:
        # Adjust indices to account for batch concatenation
        for i, pairs in enumerate(neg_fig_pairs):
            neg_fig_pairs[i] = pairs + 2 * i  # Each item adds 2 images (anchor + positive)
    
    pos_fig_pairs = torch.cat(pos_fig_pairs) if pos_fig_pairs else None
    neg_fig_pairs = torch.cat(neg_fig_pairs) if neg_fig_pairs else None
    
    return {
        'images': all_images,
        'n': n,
        'pos_patents': pos_patents,
        'neg_patents': neg_patents,
        'pos_fig_pairs': pos_fig_pairs,
        'neg_fig_pairs': neg_fig_pairs
    }
