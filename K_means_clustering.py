import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_clustering(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False
        
        df = pd.read_csv(input_path)
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_clusters = 4 * X.shape[1] - 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled) + 1
        out = pd.DataFrame({'id': ids, 'label': labels})
        out.to_csv(output_path, index=False)
        print(f"Successfully processed {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

if __name__ == "__main__":
    datasets = [
        ("./public_data.csv", "./public_submission.csv"),
        ("./private_data.csv", "./private_submission.csv")
    ]
    
    processed_count = 0
    for inp, outp in datasets:
        if run_clustering(inp, outp):
            processed_count += 1
    
    print(f"Clustering done; {processed_count}/{len(datasets)} files processed successfully.")
