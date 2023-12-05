import pytest
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox.acquisition_function.acquisition import (
    EI,
    EIC
)

class ConfigurationMock:
    def __init__(self, values=None):
        self.values = values
        self.configuration_space = ConfigurationSpace()

    def get_array(self):
        return self.values
    
class MockModel:
    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))]).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))]
        ).reshape((-1, 1))

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def acquisition_function(model):
    return EI(model)

# --------------------------------------------------------------
# Test AbstractAcquisitionFunction
# --------------------------------------------------------------

def test_update_model_and_eta(model, acquisition_function):
    model = "abc"
    assert acquisition_function.eta is None
    acquisition_function.update(model=model, eta=0.1)
    assert acquisition_function.model == model
    assert acquisition_function.eta == 0.1


def test_update_with_kwargs(acquisition_function):
    acquisition_function.update(model="abc", eta=0.0, other="hi there:)")
    assert acquisition_function.model == "abc"


def test_update_without_required(acquisition_function):
    with pytest.raises(
        TypeError,
    ):
        acquisition_function.update(other=None)

# --------------------------------------------------------------
# Test EI
# --------------------------------------------------------------

def test_ei_init(acquisition_function):
    ei = acquisition_function
    assert ei.long_name == 'Expected Improvement'
    assert abs(ei.par) < 1e-5

def test_ei_1x1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


# 测试多点输入（N维）
def test_ei_NxD(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.0)
    assert np.isclose(acq[1][0], 0.9002060113671223)
    assert np.isclose(acq[2][0], 0.3989422804014327)

def test_ei_zero_variance(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    X = np.array([ConfigurationMock([0.0])])
    acq = np.array(ei(X))
    assert np.isclose(acq[0][0], 0.0)

def test_ei_Nx1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001]),
        ConfigurationMock([1.0]),
        ConfigurationMock([2.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.9999)
    assert np.isclose(acq[1][0], 0.3989422804014327)
    assert np.isclose(acq[2][0], 0.19964122837424575)

def test_eic(mock_model):
    eic = EIC(model=mock_model, constraint_models=[mock_model], par=0.0)
    X_test = np.array([[0.1, 0.2], [0.2, 0.3]])
    eic_values = eic(X_test)
    assert eic_values.shape == (2, 1)
    
# --------------------------------------------------------------
# Test EIC
# --------------------------------------------------------------

