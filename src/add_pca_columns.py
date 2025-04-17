import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def process_clients_with_pca(clients_paths, cn_measures_type_1, cn_measures_type_2, output_folder):
    def calculate_local_covariance(df, cn_measures):
        centrality_data = df[cn_measures].fillna(0)
        scaler = StandardScaler()
        centrality_data_std = scaler.fit_transform(centrality_data)
        covariance_matrix = np.cov(centrality_data_std, rowvar=False)
        return covariance_matrix, scaler.mean_, scaler.scale_

    def apply_local_pca(df, cn_measures, n_components=2):
        centrality_data = df[cn_measures].fillna(0)
        scaler = StandardScaler()
        centrality_data_std = scaler.fit_transform(centrality_data)
        pca = PCA(n_components=n_components)
        centrality_data_pca = pca.fit_transform(centrality_data_std)
        explained_variance = pca.explained_variance_ratio_
        return explained_variance

    def apply_global_pca(df, cn_measures, mean, scale, principal_components):
        centrality_data = df[cn_measures].fillna(0)
        centrality_data_std = (centrality_data - mean) / scale
        centrality_data_pca = np.dot(centrality_data_std, principal_components)
        pca_columns = [
            f'pca_{i+1}' for i in range(principal_components.shape[1])]
        centrality_data_pca_df = pd.DataFrame(
            centrality_data_pca, columns=pca_columns, index=df.index)
        return centrality_data_pca_df

    local_covariances = []
    local_means = []
    local_scales = []
    local_explained_variances = []

    for i, client_path in enumerate(clients_paths):
        df = pd.read_parquet(client_path)
        cn_measures = cn_measures_type_2 if i < 5 else cn_measures_type_1
        client_cn_measures = [
            measure for measure in cn_measures if measure in df.columns]
        if len(client_cn_measures) == 0:
            print(f"No centrality measures found in client {i} data.")
            continue

        covariance_matrix, mean, scale = calculate_local_covariance(
            df, client_cn_measures)
        local_covariances.append(covariance_matrix)
        local_means.append(mean)
        local_scales.append(scale)
        explained_variance = apply_local_pca(df, client_cn_measures)
        local_explained_variances.append(explained_variance)

    global_covariance_matrix = np.mean(local_covariances, axis=0)
    eigen_values, eigen_vectors = np.linalg.eigh(global_covariance_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    n_components = 2
    principal_components = eigen_vectors[:, :n_components]
    explained_variances = eigen_values[:n_components] / np.sum(eigen_values)

    os.makedirs(output_folder, exist_ok=True)
    global_explained_variances = []

    for i, client_path in enumerate(clients_paths):
        df = pd.read_parquet(client_path)
        cn_measures = cn_measures_type_2 if i < 5 else cn_measures_type_1
        client_cn_measures = [
            measure for measure in cn_measures if measure in df.columns]
        if len(client_cn_measures) == 0:
            print(f"No centrality measures found in client {i} data.")
            continue
        centrality_data_pca_df = apply_global_pca(
            df, client_cn_measures, local_means[i], local_scales[i], principal_components)
        df_pca = pd.concat(
            [df, centrality_data_pca_df], axis=1)
        if i < len(clients_paths) - 1:
            output_path = os.path.join(
                output_folder, f'client_{i}_pca.parquet')
        else:
            output_path = os.path.join(output_folder, 'test_pca.parquet')
        df_pca.to_parquet(output_path)
        print(
            f'Processed PCA for {"client" if i < len(clients_paths) - 1 else "test"} {i}, saved to {output_path}')

        centrality_data_std = (df[client_cn_measures].fillna(
            0) - local_means[i]) / local_scales[i]
        global_pca_transformed = np.dot(
            centrality_data_std, principal_components)
        global_explained_variance = np.var(
            global_pca_transformed, axis=0) / np.var(centrality_data_std, axis=0).sum()
        global_explained_variances.append(global_explained_variance)

    plt.figure(figsize=(20, 12))

    for i, explained_variance in enumerate(local_explained_variances):
        plt.subplot(4, 4, i + 1)
        plt.bar(range(1, len(explained_variance) + 1),
                explained_variance, alpha=0.7, label='Local PCA')
        plt.bar(range(1, len(global_explained_variances[i]) + 1),
                global_explained_variances[i], alpha=0.7, label='Global PCA')
        plt.title(f'Client {i} Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.legend()

    plt.tight_layout()
    plt.show()


def process_clients_with_grouped_pca(feature_groups, output_folder, n_components=2, client_cn_measures=None):
    def calculate_local_pca(df, cn_measures, n_components=2):
        existing_measures = [
            measure for measure in cn_measures if measure in df.columns]

        if not existing_measures:
            raise ValueError(
                "No valid centrality measures found in DataFrame columns.")

        centrality_data = df[existing_measures].fillna(0)
        scaler = StandardScaler()
        centrality_data_std = scaler.fit_transform(centrality_data)
        pca = PCA(n_components=n_components)
        centrality_data_pca = pca.fit_transform(centrality_data_std)
        explained_variance = pca.explained_variance_ratio_
        return centrality_data_pca, explained_variance, scaler.mean_, scaler.scale_, pca.components_

    def calculate_local_covariance(pca_results):
        return np.cov(pca_results, rowvar=False)

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

    for group_id, (unique_feature_set, clients) in enumerate(feature_groups.items(), 1):
        for client_path in clients:
            print(client_path)
            df = pd.read_parquet(client_path)

            client_cn_measures = [
                measure for measure in unique_feature_set if measure in df.columns
            ]

            if len(client_cn_measures) == 0:
                print(f"No centrality measures found in client {client_path}.")
                continue

            centrality_data_pca, explained_variance, mean, scale, pca_components = calculate_local_pca(
                df, client_cn_measures, n_components)
            all_local_pca_results.append(centrality_data_pca)

            local_explained_variances[client_path] = explained_variance

            local_covariance_matrix = calculate_local_covariance(
                centrality_data_pca)
            local_covariances.append(local_covariance_matrix)

            pca_columns = [f'pca_{i+1}' for i in range(n_components)]
            local_pca_df = pd.DataFrame(
                centrality_data_pca, columns=pca_columns, index=df.index)
            df = pd.concat(
                [df, local_pca_df], axis=1)

            client_dfs[client_path] = df

    global_principal_components = apply_global_pca(local_covariances)

    global_explained_variances = {}
    scaler_post_pca = StandardScaler()

    for client_path, df in client_dfs.items():
        local_pca_data = df[[f'pca_{i+1}' for i in range(n_components)]].values

        global_pca_transformed = np.dot(
            local_pca_data, global_principal_components)

        global_pca_transformed_std = scaler_post_pca.fit_transform(
            global_pca_transformed)

        pca_columns = [f'global_pca_{j+1}' for j in range(n_components)]
        global_pca_df = pd.DataFrame(
            global_pca_transformed_std, columns=pca_columns, index=df.index)

        explained_variance = np.var(global_pca_df.values, axis=0)
        explained_variance /= np.sum(explained_variance)
        global_explained_variances[client_path] = explained_variance

        final_df = pd.concat([df.drop(
            columns=[f'pca_{i+1}' for i in range(n_components)]), global_pca_df], axis=1)

        output_path = os.path.join(
            output_folder, f'{os.path.basename(client_path).split(".")[0]}.parquet')
        final_df.to_parquet(output_path)
        print(
            f'Processed federated PCA for client {client_path}, saved to {output_path}')

        client_dfs[client_path] = final_df

        local_reconstructed = np.dot(
            local_pca_data, global_principal_components.T)

        local_reconstructed_std = scaler_post_pca.transform(
            local_reconstructed)

        reconstruction_errors_local[client_path] = mean_squared_error(
            scaler_post_pca.transform(local_pca_data), local_reconstructed_std)
        reconstruction_errors_federated[client_path] = mean_squared_error(
            scaler_post_pca.transform(local_pca_data), global_pca_transformed_std)

        print(
            f"Client {client_path} Local PCA Reconstruction Error: {reconstruction_errors_local[client_path]}")
        print(
            f"Client {client_path} Federated PCA Reconstruction Error: {reconstruction_errors_federated[client_path]}")

    return {
        'reconstruction_errors_local': reconstruction_errors_local,
        'reconstruction_errors_federated': reconstruction_errors_federated,
    }, pca_columns


def evaluate_pca_results(clients_paths):
    for client_idx, client_path in enumerate(clients_paths):
        df = pd.read_parquet(client_path)

        if 'global_pca_1' not in df.columns or 'global_pca_2' not in df.columns or 'Label' not in df.columns:
            print(
                f"Skipping client {client_idx} as required columns are missing.")
            continue

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['global_pca_1'], df['global_pca_2'], c=df['Label'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Label')
        plt.title(f'PCA Plot for Client {client_idx}')
        plt.xlabel('1st Principal Component (global_pca_1)')
        plt.ylabel('2nd Principal Component (global_pca_2)')
        plt.show()

        kmeans = KMeans(n_clusters=len(df['Label'].unique()))
        kmeans.fit(df[['global_pca_1', 'global_pca_2']])
        cluster_labels = kmeans.labels_
        sil_score = silhouette_score(
            df[['global_pca_1', 'global_pca_2']], cluster_labels)
        print(f"Silhouette Score for client {client_idx}: {sil_score}")

        X_train, X_test, y_train, y_test = train_test_split(
            df[['global_pca_1', 'global_pca_2']], df['Label'], test_size=0.3, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(
            f"Train/Test Split Classification performance for client {client_idx}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("\n" + "="*50 + "\n")

        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds)
        cv_accuracy = cross_val_score(
            clf, df[['global_pca_1', 'global_pca_2']], df['Label'], cv=skf, scoring='accuracy')
        cv_precision = cross_val_score(
            clf, df[['global_pca_1', 'global_pca_2']], df['Label'], cv=skf, scoring='precision_weighted')
        cv_recall = cross_val_score(
            clf, df[['global_pca_1', 'global_pca_2']], df['Label'], cv=skf, scoring='recall_weighted')
        cv_f1 = cross_val_score(
            clf, df[['global_pca_1', 'global_pca_2']], df['Label'], cv=skf, scoring='f1_weighted')

        print(
            f"Cross-Validation (k={k_folds}) performance for client {client_idx}:")
        print(
            f"CV Accuracy: {np.mean(cv_accuracy)} (+/- {np.std(cv_accuracy)})")
        print(
            f"CV Precision: {np.mean(cv_precision)} (+/- {np.std(cv_precision)})")
        print(f"CV Recall: {np.mean(cv_recall)} (+/- {np.std(cv_recall)})")
        print(f"CV F1 Score: {np.mean(cv_f1)} (+/- {np.std(cv_f1)})")
        print("\n" + "="*50 + "\n")


def process_clients_with_grouped_pca_rmse(feature_groups, df_list, output_folder, n_components=2, client_cn_measures=None):
    def calculate_local_pca(df, cn_measures, n_components=2):
        existing_measures = [
            measure for measure in cn_measures if measure in df.columns]

        if not existing_measures:
            raise ValueError(
                "No valid centrality measures found in DataFrame columns.")

        centrality_data = df[existing_measures].fillna(0)
        scaler = StandardScaler()
        centrality_data_std = scaler.fit_transform(centrality_data)
        pca = PCA(n_components=n_components)
        centrality_data_pca = pca.fit_transform(centrality_data_std)
        explained_variance = pca.explained_variance_ratio_
        return centrality_data_pca, explained_variance, scaler.mean_, scaler.scale_, pca.components_

    def calculate_local_covariance(pca_results):
        return np.cov(pca_results, rowvar=False)

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

    for group_id, (unique_feature_set, clients) in enumerate(feature_groups.items(), 1):
        for client_path in clients:
            print(client_path)
            df = df_list[client_path]
            # df = pd.read_parquet(client_path)

            client_cn_measures = [
                measure for measure in unique_feature_set if measure in df.columns
            ]

            if len(client_cn_measures) == 0:
                print(f"No centrality measures found in client {client_path}.")
                continue

            centrality_data_pca, explained_variance, mean, scale, pca_components = calculate_local_pca(
                df, client_cn_measures, n_components)
            all_local_pca_results.append(centrality_data_pca)

            df.drop(columns=client_cn_measures, inplace=True)

            local_explained_variances[client_path] = explained_variance

            local_covariance_matrix = calculate_local_covariance(
                centrality_data_pca)
            local_covariances.append(local_covariance_matrix)

            pca_columns = [f'pca_{i+1}' for i in range(n_components)]
            local_pca_df = pd.DataFrame(
                centrality_data_pca, columns=pca_columns, index=df.index)
            df = pd.concat([df, local_pca_df], axis=1)

            client_dfs[client_path] = df

    global_principal_components = apply_global_pca(local_covariances)

    global_explained_variances = {}
    scaler_post_pca = StandardScaler()  #

    for client_path, df in client_dfs.items():
        local_pca_data = df[[f'pca_{i+1}' for i in range(n_components)]].values

        global_pca_transformed = np.dot(
            local_pca_data, global_principal_components)

        global_pca_transformed_std = scaler_post_pca.fit_transform(
            global_pca_transformed)

        pca_columns = [f'global_pca_{j+1}' for j in range(n_components)]
        global_pca_df = pd.DataFrame(
            global_pca_transformed_std, columns=pca_columns, index=df.index)

        final_df = pd.concat([df.drop(
            columns=[f'pca_{i+1}' for i in range(n_components)]), global_pca_df], axis=1)

        output_path = os.path.join(
            output_folder, f'{os.path.basename(client_path).split(".")[0]}.parquet')
        final_df.to_parquet(output_path)
        print(
            f'Processed federated PCA for client {client_path}, saved to {output_path}')

        client_dfs[client_path] = final_df

        local_reconstructed = np.dot(
            local_pca_data, global_principal_components.T)

        local_reconstructed_std = scaler_post_pca.transform(
            local_reconstructed)

        rmse_local = np.sqrt(mean_squared_error(
            scaler_post_pca.transform(local_pca_data), local_reconstructed_std))
        rmse_federated = np.sqrt(mean_squared_error(
            scaler_post_pca.transform(local_pca_data), global_pca_transformed_std))

        reconstruction_errors_local[client_path] = rmse_local
        reconstruction_errors_federated[client_path] = rmse_federated

        print(f"Client {client_path} Local PCA RMSE: {rmse_local}")
        print(f"Client {client_path} Federated PCA RMSE: {rmse_federated}")

    return client_dfs, {
        'reconstruction_errors_local': reconstruction_errors_local,
        'reconstruction_errors_federated': reconstruction_errors_federated,
    }, pca_columns
