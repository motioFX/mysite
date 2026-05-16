import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Load the dataset with multi-index header
    file_path = 'global_markets_ohlcv_1y-1.csv'
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)

    # Extract the 'Close' prices for all assets
    print("Extracting 'Close' prices...")
    close_prices = df.xs('Close', level=1, axis=1)

    # Handle missing values
    # Forward fill missing values first (e.g. weekends/holidays)
    close_prices = close_prices.ffill()
    # Drop any remaining rows with missing values (e.g., beginning of dataset)
    close_prices = close_prices.dropna()

    print(f"Data shape after handling missing values: {close_prices.shape}")

    # Standardize the data
    # PCA is sensitive to scaling, so we scale features to have mean=0 and variance=1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Perform PCA
    print("Performing PCA...")
    # Keep components that explain a good amount of variance, or just compute all
    pca = PCA()
    pca.fit(scaled_data)

    # Display the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    print("\nPCA Results:")
    print("------------")
    for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"Principal Component {i+1}:")
        print(f"  Explained Variance Ratio: {ev:.4f} ({ev*100:.2f}%)")
        print(f"  Cumulative Variance:      {cv:.4f} ({cv*100:.2f}%)")
        if i >= 4: # just show top 5 for brevity in stdout
            print("  ...")
            break

    # Output PCA Loadings for PC1 and PC2
    print("\nPCA Loadings (Eigenvectors) for PC1 and PC2:")
    print("--------------------------------------------")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=close_prices.columns
    )
    # Print the sorted loadings for PC1 and PC2 to see which assets contribute most
    print("Top contributors to PC1 (absolute value):")
    pc1_sorted = loadings['PC1'].abs().sort_values(ascending=False)
    for asset in pc1_sorted.index:
        print(f"  {asset}: {loadings.loc[asset, 'PC1']:.4f}")

    print("\nTop contributors to PC2 (absolute value):")
    pc2_sorted = loadings['PC2'].abs().sort_values(ascending=False)
    for asset in pc2_sorted.index:
        print(f"  {asset}: {loadings.loc[asset, 'PC2']:.4f}")

    # Example: How to get the transformed data (principal components)
    pca_data = pca.transform(scaled_data)
    pca_df = pd.DataFrame(
        pca_data,
        index=close_prices.index,
        columns=[f'PC{i+1}' for i in range(pca_data.shape[1])]
    )

    print("\nFirst 5 rows of PCA transformed data:")
    print(pca_df.iloc[:, :3].head())

if __name__ == '__main__':
    main()
