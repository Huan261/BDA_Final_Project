import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np

def run_clustering(input_path, output_path, method=1):
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False
        
        df = pd.read_csv(input_path)
        
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:].values
        n_dim = X.shape[1]
        n_clusters = 4 * n_dim - 1
        
        # Method selection
        if method == 1:
            # K-Means
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled) + 1
        elif method == 2:
            # Method 2: K-Means with PCA and RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_dim, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca) + 1
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
        elif method == 4:
            # Method 4: GMM with PCA and RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_dim, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            labels = gmm.fit_predict(X_pca) + 1
        elif method == 5:
            # Method 5: Hybrid GMM (StandardScaler, tied covariance, augment with dim3+dim4 if possible)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if n_dim >= 4:
                # Augment with sum of 3rd and 4th dimensions (indices 2 and 3)
                dim34_sum = (X_scaled[:, 2] + X_scaled[:, 3]).reshape(-1, 1)
                X_processed = np.hstack((X_scaled, dim34_sum))
            else:
                X_processed = X_scaled
            gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=42)
            labels = gmm.fit_predict(X_processed) + 1
        elif method == 6:
            # Method 6: Advanced GMM with PowerTransformer and sophisticated feature engineering
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            X_transformed = scaler.fit_transform(X)
            
            # Advanced feature augmentation: multiple approaches for dims 2&3
            if n_dim >= 3:
                dim2 = X_transformed[:, 1].reshape(-1, 1)  # 2nd dimension
                dim3 = X_transformed[:, 2].reshape(-1, 1)  # 3rd dimension
                
                # Create multiple derived features
                dim23_sum = (dim2 + dim3)  # Sum
                dim23_diff = (dim2 - dim3)  # Difference
                dim23_product = (dim2 * dim3)  # Product
                
                X_processed = np.hstack((X_transformed, dim2, dim3, dim23_sum, dim23_diff, dim23_product))
            else:
                X_processed = X_transformed
            
            # Try multiple covariance types and select best based on BIC
            best_gmm = None
            best_bic = np.inf
            
            for cov_type in ['tied', 'full', 'diag']:
                for seed in [42, 11, 99]:  # Multiple seeds for robustness
                    gmm = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type=cov_type,
                        random_state=seed,
                        n_init=30,
                        max_iter=600,
                        tol=1e-7,
                        reg_covar=1e-6  # Small regularization for numerical stability
                    )
                    gmm.fit(X_processed)
                    bic = gmm.bic(X_processed)
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_gmm = gmm
            
            labels = best_gmm.predict(X_processed) + 1
        else:
            print(f"Error: Invalid method {method}. Please choose between 1-6.")
            return False
        
        out = pd.DataFrame({'id': ids, 'label': labels})
        out.to_csv(output_path, index=False)
        print(f"Successfully processed {input_path} -> {output_path} using method {method}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # Method selection
    print("Available clustering methods:")
    print("1. K-Means (StandardScaler)")
    print("2. K-Means with PCA (RobustScaler)")
    print("3. K-Means with Feature Augmentation (StandardScaler, emphasizing dims 2&3)")
    print("4. GMM with PCA and RobustScaler")
    print("5. Hybrid GMM (StandardScaler, tied covariance, augment with dim3+dim4 if possible)")
    print("6. GMM with PowerTransformer and tied covariance")
    
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
