from src.utils import (
    load_df,
    load_config,
    NumpyEncoder,
    plot_confusion_matrix,
    calculate_fpr_fnr_with_global,
)
import pytest
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for plotting


# --- load_df tests ---


def test_load_df_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)
    loaded = load_df(str(csv_file), raw_type="csv")
    pd.testing.assert_frame_equal(loaded, df)


def test_load_df_parquet(tmp_path):
    # skip if no parquet engine
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    pq_file = tmp_path / "test.parquet"
    df.to_parquet(pq_file)
    loaded = load_df(str(pq_file), raw_type="parquet")
    pd.testing.assert_frame_equal(loaded, df)


def test_load_df_invalid_type(tmp_path):
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("nope")
    assert load_df(str(dummy), raw_type="json") is None


# --- load_config tests ---

def test_load_config_simple(monkeypatch):
    calls = {}

    def fake_initialize(version_base, config_path):
        class Ctx:
            def __enter__(self_inner):
                calls['initialize'] = dict(
                    version_base=version_base, config_path=config_path)

            def __exit__(self_inner, exc_type, exc, tb):
                pass
        return Ctx()

    def fake_compose(config_name):
        calls['compose'] = config_name
        return {"cfg": config_name}

    monkeypatch.setattr("src.utils.initialize", fake_initialize)
    monkeypatch.setattr("src.utils.compose", fake_compose)

    # simple config_name (no slash)
    out = load_config("myconf")
    assert out == {"cfg": "myconf"}
    assert calls['initialize']['config_path'] == "../conf"
    assert calls['compose'] == "myconf"


def test_load_config_subfolder(monkeypatch):
    calls = {}

    def fake_initialize(version_base, config_path):
        class Ctx:
            def __enter__(self_inner):
                calls['initialize'] = dict(
                    version_base=version_base, config_path=config_path)

            def __exit__(self_inner, exc_type, exc, tb):
                pass
        return Ctx()

    monkeypatch.setattr("src.utils.initialize", fake_initialize)
    monkeypatch.setattr("src.utils.compose", lambda config_name: config_name)

    out = load_config("group/sub1/mycfg")
    assert out == "mycfg"
    assert calls['initialize']['config_path'] == "../conf/group/sub1"


# --- NumpyEncoder tests ---

def test_numpy_encoder_types():
    data = {
        "i": np.int32(7),
        "f": np.float64(2.5),
        "arr": np.array([4, 5, 6]),
        "norm": "ok"
    }
    s = json.dumps(data, cls=NumpyEncoder)
    loaded = json.loads(s)
    assert isinstance(loaded['i'], int)
    assert isinstance(loaded['f'], float)
    assert loaded['arr'] == [4, 5, 6]
    assert loaded['norm'] == "ok"


# --- plot_confusion_matrix tests ---

def test_plot_confusion_matrix_basic(tmp_path):
    cm = np.array([[2, 1], [3, 4]])
    names = ["A", "B"]
    # no file save, no show
    fig = plot_confusion_matrix(
        cm,
        target_names=names,
        title="CM Test",
        normalized=False,
        file_path=None,
        show_figure=False
    )
    # Should return a matplotlib Figure
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "CM Test"
    # x-ticks labels should match names
    xt = [lbl.get_text() for lbl in ax.get_xticklabels()]
    assert xt == names

    # test saving to disk
    out_fp = tmp_path / "cm.png"
    fig2 = plot_confusion_matrix(
        cm,
        target_names=names,
        title="Save Test",
        normalized=True,
        file_path=str(out_fp),
        show_figure=False
    )
    assert out_fp.exists()


# --- calculate_fpr_fnr_with_global tests ---

def test_calculate_fpr_fnr_with_global_values():
    # 2-class example
    cm = np.array([[50, 10], [5, 35]])
    res = calculate_fpr_fnr_with_global(cm)

    # class 0: FP=5, TN=35, FN=10, TP=50
    assert pytest.approx(res['per_class'][0]['FPR'], rel=1e-6) == 5 / (5 + 35)
    assert pytest.approx(res['per_class'][0]['FNR'],
                         rel=1e-6) == 10 / (50 + 10)

    # class 1: FP=10, TN=50, FN=5, TP=35
    assert pytest.approx(res['per_class'][1]['FPR'],
                         rel=1e-6) == 10 / (10 + 50)
    assert pytest.approx(res['per_class'][1]['FNR'], rel=1e-6) == 5 / (5 + 35)

    # global: FP_total=15, TN_total=85, FN_total=15, TP_total=85
    assert pytest.approx(res['global']['FPR'], rel=1e-6) == 15 / (15 + 85)
    assert pytest.approx(res['global']['FNR'], rel=1e-6) == 15 / (15 + 85)


def test_calculate_fpr_fnr_with_global_edge_zero():
    # zero confusion matrix
    cm = np.zeros((1, 1), dtype=int)
    res = calculate_fpr_fnr_with_global(cm)
    assert res['per_class'][0]['FPR'] is None
    assert res['per_class'][0]['FNR'] is None
    assert res['global']['FPR'] is None
    assert res['global']['FNR'] is None
