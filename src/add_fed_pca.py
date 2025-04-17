import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def process_clients_with_grouped_pca_rmse(client_names, features_list, df_list, output_folder, n_components=2):
    # Function to calculate local PCA from a single client's DataFrame.
    def calculate_local_pca(df, cn_measures, n_components=2):
        # Keep only the centrality measures that actually exist in the DataFrame.
        existing_measures = [
            measure for measure in cn_measures if measure in df.columns]
        if not existing_measures:
            raise ValueError(
                "No valid centrality measures found in DataFrame columns.")

        # Prepare data by filling missing values.
        centrality_data = df[existing_measures].fillna(0)
        scaler = StandardScaler()
        centrality_data_std = scaler.fit_transform(centrality_data)
        pca = PCA(n_components=n_components)
        centrality_data_pca = pca.fit_transform(centrality_data_std)
        explained_variance = pca.explained_variance_ratio_
        return centrality_data_pca, explained_variance, scaler.mean_, scaler.scale_, pca.components_

    # Calculate covariance of the PCA-transformed data.
    def calculate_local_covariance(pca_results):
        return np.cov(pca_results, rowvar=False)

    # Combine local covariance matrices to compute the global principal components.
    def apply_global_pca(local_covariances):
        global_covariance_matrix = np.mean(local_covariances, axis=0)
        eigen_values, eigen_vectors = np.linalg.eigh(global_covariance_matrix)
        sorted_indices = np.argsort(eigen_values)[::-1]
        global_principal_components = eigen_vectors[:,
                                                    sorted_indices][:, :n_components]
        return global_principal_components

    os.makedirs(output_folder, exist_ok=True)

    all_local_pca_results = []
    local_covariances = []
    client_dfs = {}
    local_explained_variances = {}

    reconstruction_errors_local = {}
    reconstruction_errors_federated = {}

    # Process each client individually.
    for client in client_names:
        print(f"Processing client: {client}")
        # Get the client's DataFrame and feature list.
        df = df_list[client]
        client_features = features_list[client]

        # Identify centrality measures present in the client's DataFrame.
        client_cn_measures = [
            measure for measure in client_features if measure in df.columns]
        if len(client_cn_measures) == 0:
            print(f"No centrality measures found for client {client}.")
            continue

        # Compute the local PCA for this client.
        try:
            centrality_data_pca, explained_variance, mean, scale, pca_components = calculate_local_pca(
                df, client_cn_measures, n_components
            )
        except ValueError as e:
            print(f"Error processing client {client}: {e}")
            continue

        all_local_pca_results.append(centrality_data_pca)
        local_explained_variances[client] = explained_variance

        # Compute covariance matrix of the local PCA results.
        local_covariance_matrix = calculate_local_covariance(
            centrality_data_pca)
        local_covariances.append(local_covariance_matrix)

        # Create a temporary DataFrame with local PCA results.
        pca_columns = [f'pca_{i+1}' for i in range(n_components)]
        local_pca_df = pd.DataFrame(
            centrality_data_pca, columns=pca_columns, index=df.index)
        df = pd.concat([df, local_pca_df], axis=1)
        client_dfs[client] = df

    # Compute global principal components using the average of local covariances.
    global_principal_components = apply_global_pca(local_covariances)

    scaler_post_pca = StandardScaler()

    # Process each client's DataFrame for global transformation.
    for client, df in client_dfs.items():
        local_pca_data = df[[f'pca_{i+1}' for i in range(n_components)]].values

        # Apply global PCA using the obtained global principal components.
        global_pca_transformed = np.dot(
            local_pca_data, global_principal_components)
        global_pca_transformed_std = scaler_post_pca.fit_transform(
            global_pca_transformed)

        pca_columns = [f'global_pca_{j+1}' for j in range(n_components)]
        global_pca_df = pd.DataFrame(
            global_pca_transformed_std, columns=pca_columns, index=df.index)

        # Replace local PCA columns with global PCA columns in the DataFrame.
        final_df = pd.concat([df.drop(
            columns=[f'pca_{i+1}' for i in range(n_components)]), global_pca_df], axis=1)

        # Save the processed DataFrame to the output folder.
        output_path = os.path.join(output_folder, f'{client}.parquet')
        final_df.to_parquet(output_path)
        print(
            f'Processed federated PCA for client {client}, saved to {output_path}')

        client_dfs[client] = final_df

        # Compute reconstruction of the local PCA data using the global principal components.
        local_reconstructed = np.dot(
            local_pca_data, global_principal_components.T)
        local_reconstructed_std = scaler_post_pca.transform(
            local_reconstructed)

        # Calculate RMSE errors for reconstruction.
        rmse_local = np.sqrt(mean_squared_error(
            scaler_post_pca.transform(local_pca_data), local_reconstructed_std))
        rmse_federated = np.sqrt(mean_squared_error(
            scaler_post_pca.transform(local_pca_data), global_pca_transformed_std))

        reconstruction_errors_local[client] = rmse_local
        reconstruction_errors_federated[client] = rmse_federated

        print(f"Client {client} Local PCA RMSE: {rmse_local}")
        print(f"Client {client} Federated PCA RMSE: {rmse_federated}")

    return {
        'reconstruction_errors_local': reconstruction_errors_local,
        'reconstruction_errors_federated': reconstruction_errors_federated,
    }, pca_columns
