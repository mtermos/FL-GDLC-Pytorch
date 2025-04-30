import json
import numpy as np
import os


def calculate_df_properties(df, G, label_col, processed_dir, name):
    total_count = len(df)

    properties = {
        "name": name,
        "length": total_count,
    }

    num_benign = len(df[df[label_col] == 0])
    num_attack = len(df[df[label_col] == 1])

    properties["num_benign"] = num_benign
    properties["percentage_of_benign_records"] = (
        (num_benign * 100)/total_count)

    properties["num_attack"] = num_attack
    properties["percentage_of_attack_records"] = (
        (num_attack * 100)/total_count)

    properties["attacks"] = list(df["Attack"].unique())

    properties["number_of_nodes"] = G.number_of_nodes()
    properties["number_of_edges"] = G.number_of_edges()

    properties["average_degree"] = np.mean(list(dict(G.degree()).values()))

    with open(os.path.join(processed_dir, name + '_properties.json'), 'w') as f:
        json.dump(properties, f)
