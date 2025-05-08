import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data.calculate_df_properties import calculate_df_properties


@pytest.fixture
def sample_df():
    # 6 total rows: 4 benign (label 0) and 2 attack (label 1)
    data = {
        "label": [0, 1, 0, 1, 0, 0],
        "class": ["A", "B", "A", "C", "B", "C"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_graph():
    # simple 3-node path: nodes 1-2-3
    G = nx.path_graph(3)
    return G


def test_calculate_df_properties_writes_correct_json(sample_df, sample_graph, tmp_path):
    name = "testset"
    label_col = "label"
    class_col = "class"
    processed_dir = str(tmp_path)

    # call under test
    calculate_df_properties(
        df=sample_df,
        G=sample_graph,
        label_col=label_col,
        class_col=class_col,
        processed_dir=processed_dir,
        name=name
    )

    # path to output
    json_path = os.path.join(processed_dir, name + "_properties.json")
    assert os.path.isfile(json_path), "JSON file was not created"

    # load and verify contents
    with open(json_path, "r") as f:
        props = json.load(f)

    # basic counts
    assert props["name"] == name
    assert props["length"] == 6

    # benign vs attack
    assert props["num_benign"] == 4
    assert pytest.approx(
        props["percentage_of_benign_records"], rel=1e-6) == (4 * 100 / 6)
    assert props["num_attack"] == 2
    assert pytest.approx(
        props["percentage_of_attack_records"], rel=1e-6) == (2 * 100 / 6)

    # unique classes (order may vary)
    assert set(props["attacks"]) == {"A", "B", "C"}

    # graph properties
    assert props["number_of_nodes"] == 3
    assert props["number_of_edges"] == 2

    # average degree: degrees are [1,2,1] so mean = 4/3
    expected_avg_degree = np.mean([1, 2, 1])
    assert pytest.approx(props["average_degree"],
                         rel=1e-6) == expected_avg_degree


def test_empty_df_raises_or_handles_zero_division(tmp_path, sample_graph):
    # If df is empty, total_count == 0, division by zero would occur.
    # Decide on desired behavior: e.g. raise ZeroDivisionError.
    df_empty = pd.DataFrame({"label": [], "class": []})
    with pytest.raises(ZeroDivisionError):
        calculate_df_properties(
            df=df_empty,
            G=sample_graph,
            label_col="label",
            class_col="class",
            processed_dir=str(tmp_path),
            name="empty"
        )
