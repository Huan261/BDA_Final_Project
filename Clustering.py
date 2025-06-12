import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(X, labels, output_dir, method_name, dataset_name):
    """
    Visualize clustering results with:
    1. PCA projection
    2. Pairwise dimension plots (1&2, 2&3, 3&4)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set seaborn style properly
    sns.set_theme(style="whitegrid")
    
    # 1. PCA Visualization
    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'PCA Projection - {dataset_name} ({method_name})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_pca_{method_name}.png'))
    
    # 2. Dimension Pairs Visualization
    dim_pairs = [(0, 1), (1, 2), (2, 3)]  # 1&2, 2&3, 3&4 (using 0-indexing)
    
    for i, j in dim_pairs:
        # Skip if dimensions don't exist in the data
        if max(i, j) >= X.shape[1]:
            continue
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, i], X[:, j], c=labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Dimensions {i+1} & {j+1} - {dataset_name} ({method_name})')
        plt.xlabel(f'Dimension {i+1}')
        plt.ylabel(f'Dimension {j+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_dim_{i+1}_{j+1}_{method_name}.png'))
    
    # Close all open figures to free memory
    plt.close('all')

def run_clustering(input_path, output_path, method=1):
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False
        
        # Determine dataset name (public or private)
        dataset_name = "public" if "public" in input_path else "private"
        
        df = pd.read_csv(input_path)
        
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:].values
        n_dim = X.shape[1]
        n_clusters = 4 * n_dim - 1
        
        # Get method name for visualization
        method_names = {
            1: "KMeans_StandardScaler",
            2: "KMeans_PCA_RobustScaler",
            3: "KMeans_FeatureAugmentation",
            4: "GMM_PCA_RobustScaler",
            5: "GMM_TiedCovariance",
            6: "GMM_PowerTransformer"
        }
        method_name = method_names.get(method, f"Method_{method}")
        
        # Create visualization directory
        vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
        
        # For storing processed data to visualize
        X_for_vis = X.copy()  # Default to original data, may be overridden
        
        if method == 1:
            # K-Means
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled) + 1
            X_for_vis = X_scaled  # Use scaled data for visualization
        elif method == 2:
            # Method 2: K-Means with PCA and RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_dim, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca) + 1
            X_for_vis = X_pca  # Use PCA-transformed data for visualization
        elif method == 3:
            # Method 3: K-Means with Feature Augmentation (StandardScaler)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if n_dim >= 3: # Ensure dimensions at index 1 and 2 exist for augmentation
                # Emphasize 2nd and 3rd dimensions (indices 1 and 2)
                X_dim_emphasized = X_scaled[:, [1, 2]] 
                X_processed = np.hstack((X_scaled, X_dim_emphasized))
            else: # Fallback if less than 3 dimensions
                X_processed = X_scaled
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_processed) + 1
            X_for_vis = X_processed  # Use processed data for visualization
        elif method == 4:
            # Method 4: GMM with PCA and RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_dim, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            labels = gmm.fit_predict(X_pca) + 1
            X_for_vis = X_pca  # Use PCA-transformed data for visualization
        elif method == 5:
            # Method 5: GMM (StandardScaler, tied covariance, emphasize dimensions 2 and 3)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if n_dim >= 3:
                # Emphasize 2nd and 3rd dimensions (indices 1 and 2)
                X_dim_emphasized = X_scaled[:, [1, 2]]
                X_processed = np.hstack((X_scaled, X_dim_emphasized))
            else:
                X_processed = X_scaled
            gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=42)
            labels = gmm.fit_predict(X_processed) + 1
            X_for_vis = X_processed  # Use processed data for visualization
        elif method == 6:
            # Method 6: GMM with PowerTransformer, tied covariance, and feature augmentation
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            X_transformed = scaler.fit_transform(X)
            
            # Feature augmentation: emphasize 2nd and 3rd dimensions
            if n_dim >= 3:
                # Add emphasized 2nd and 3rd dimensions (indices 1 and 2)
                X_dim_emphasized = X_transformed[:, [1, 2]]
                X_processed = np.hstack((X_transformed, X_dim_emphasized))
            else:
                X_processed = X_transformed
            
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='tied',
                random_state=42,
                n_init=25,  
                max_iter=500,  
                tol=1e-6  
            )
            labels = gmm.fit_predict(X_processed) + 1
            X_for_vis = X_processed  # Use processed data for visualization
        else:
            print(f"Error: Invalid method {method}. Please choose between 1-6.")
            return False
        
        # Generate visualizations
        # 1. Transformed data visualizations - Shows clusters in the transformed space where clustering was performed
        #    This reveals how the algorithm "sees" the data after preprocessing (scaling, PCA, etc.)
        # visualize_clusters(X_for_vis, labels, vis_dir, method_name, dataset_name)
        
        # 2. Original data visualizations - Shows same cluster assignments in original feature space
        #    This helps with interpretability as original dimensions have meaningful real-world units and semantics
        #    It allows understanding what characteristics define each cluster in the raw data
        visualize_clusters(X, labels, vis_dir, f"{method_name}_original", dataset_name)
        
        out = pd.DataFrame({'id': ids, 'label': labels})
        out.to_csv(output_path, index=False)
        print(f"Successfully processed {input_path} -> {output_path} using method {method}")
        print(f"Visualizations saved to {vis_dir}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # Method selection
    print("Available clustering methods:")
    print("1. K-Means with StandardScaler")
    print("2. K-Means with PCA and RobustScaler")
    print("3. K-Means with feature augmentation")
    print("4. GMM with PCA and RobustScaler")
    print("5. GMM with tied covariance and feature augmentation")
    print("6. GMM with PowerTransformer, tied covariance, and feature augmentation")
    
    try:
        method = int(input("Choose a method (1-6): "))
        if method not in range(1, 7):
            print("Invalid method. Using default method 1 (K-Means).")
            method = 1
    except ValueError:
        print("Invalid input. Using default method 1 (K-Means).")
        method = 1
    
    datasets = [
        ("./public_data.csv", "./public_submission.csv"),
        ("./private_data.csv", "./private_submission.csv")
    ]
    
    processed_count = 0
    for inp, outp in datasets:
        if run_clustering(inp, outp, method):
            processed_count += 1
    
    print(f"Clustering done; {processed_count}/{len(datasets)} files processed successfully.")