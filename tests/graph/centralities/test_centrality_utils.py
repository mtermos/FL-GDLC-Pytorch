import pytest

from src.graph.centralities.centrality_utils import hm_rescale


def test_hm_rescale_simple():
    # Rescaling a simple dictionary
    values = {'a': 2, 'b': 4, 'c': 1}
    # After rescaling by the maximum (4), we expect: a=0.5, b=1.0, c=0.25
    expected = {'a': 0.5, 'b': 1.0, 'c': 0.25}
    result = hm_rescale(values)
    assert result == pytest.approx(expected)
