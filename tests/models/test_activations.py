import torch
import pytest
from src.models.activations import ACTIVATIONS


def test_activations_dict():
    """Test that ACTIVATIONS dictionary contains all expected activation functions"""
    expected_activations = ["relu", "leaky_relu", "gelu", "tanh", "none"]
    assert all(act in ACTIVATIONS for act in expected_activations)


def test_relu_activation():
    """Test ReLU activation function"""
    relu = ACTIVATIONS["relu"]()
    x = torch.tensor([-1.0, 0.0, 1.0])
    output = relu(x)
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(output, expected)


def test_leaky_relu_activation():
    """Test LeakyReLU activation function"""
    leaky_relu = ACTIVATIONS["leaky_relu"]()
    x = torch.tensor([-1.0, 0.0, 1.0])
    output = leaky_relu(x)
    expected = torch.tensor([-0.01, 0.0, 1.0])  # default negative_slope=0.01
    assert torch.allclose(output, expected)


def test_gelu_activation():
    """Test GELU activation function"""
    gelu = ACTIVATIONS["gelu"]()
    x = torch.tensor([-1.0, 0.0, 1.0])
    output = gelu(x)
    # GELU is a smooth approximation of ReLU
    assert output[0] < 0  # negative input should be negative but not zero
    assert output[1] == 0  # zero input should be zero
    assert output[2] > 0  # positive input should be positive


def test_tanh_activation():
    """Test Tanh activation function"""
    tanh = ACTIVATIONS["tanh"]()
    x = torch.tensor([-1.0, 0.0, 1.0])
    output = tanh(x)
    expected = torch.tensor([-0.7616, 0.0, 0.7616], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-4)


def test_none_activation():
    """Test that 'none' activation returns None"""
    assert ACTIVATIONS["none"] is None
