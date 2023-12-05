import pytest
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox.acquisition_function.acquisition import (
    EI,
    EIC,
    EIPS,
    LogEI,
    LPEI,
    PI,
    LCB,
    Uncertainty,
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
    
    def predict(self, X: np.ndarray):
        return self.predict_marginalized_over_instances(X)
    

class MockModelDual:
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 2))
    
class MockConstraintModel:
    def predict_marginalized_over_instances(self, X):
        m = -np.abs(np.mean(X, axis=1)).reshape((-1, 1))
        v = np.full(m.shape, 0.1)
        return m, v


@pytest.fixture
def model():
    return MockModel()

def constraint_model():
    return MockConstraintModel()

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


# --------------------------------------------------------------
# Test EIC
# --------------------------------------------------------------

@pytest.fixture
def acq_eic_single(model, constraint_model):
    return EIC(model = model, constraint_models = [constraint_model])

def acq_eic_multi(model, constraint_model):
    return EIC(model = model, constraint_models = [constraint_model, constraint_model])

def test_eic_init(acq_eic):
    assert acq_eic.long_name == 'Expected Constrained Improvement'

def test_eic_single_constraint(model, acq_eic_single):
    eic = acq_eic_single
    eic.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.5])]
    acq = eic(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.5656769359904809)

def test_eic_multi_constraint(model, acq_eic_multi):
    eic = acq_eic_multi
    eic.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = eic(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.25010115992223825)
    assert np.isclose(acq[1][0], 0.3506143218346858)
    assert np.isclose(acq[2][0], 0.39831801965532126)


# --------------------------------------------------------------
# Test EIPS
# --------------------------------------------------------------

@pytest.fixture
def model_eips():
    return MockModelDual()


@pytest.fixture
def acq_eips(model_eips):
    return EIPS(model_eips)


def test_eips_init(acq_eips):
    assert acq_eips.long_name == 'Expected Improvement per Second'

def test_eips_1xD(model_eips, acq_eips):
    eips = acq_eips
    eips.update(model=model_eips, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
    acq = eips(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_eips_fail(model_eips, acq_eips):
    eips = acq_eips
    with pytest.raises(ValueError):
        configurations = [ConfigurationMock([1.0, 1.0])]
        eips(configurations)

# --------------------------------------------------------------
# Test LogEI
# --------------------------------------------------------------

@pytest.fixture
def acq_logei(model):
    return LogEI(model)

def test_logei_init(acq_logei):
    assert acq_logei.long_name == 'Log Expected Improvement'


def test_logei_1xD(model, acq_logei):
    logei = acq_logei
    logei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    acq = logei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.6480973967332011)


def test_logei_NxD(model, acq_logei):
    logei = acq_logei
    logei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.1, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = logei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.6670107375002425)
    assert np.isclose(acq[1][0], 1.5570607606556273)
    assert np.isclose(acq[2][0], 0.6480973967332011)

# --------------------------------------------------------------
# Test LPEI
# --------------------------------------------------------------

@pytest.fixture
def acq_lpei(model):
    return LPEI(model)

def test_lpei_init(acq_lpei):
    assert acq_lpei.long_name == 'Expected Improvement with Local Penalizer'
    assert abs(acq_lpei.estimate_L) < 1e-5
    assert len(acq_lpei.batch_configs) == 0

def test_lpei_1x1(model, acq_lpei):
    lpei = acq_lpei
    lpei.update(model=model, eta=1.0)

    configurations = [ConfigurationMock([0.5])]
    batch_configs = [ConfigurationMock([0.6])]

    lpei.batch_configs = batch_configs

    acq = lpei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], -0.09414608748817546)

def test_lpei_NxD(model, acq_lpei):
    lpei = acq_lpei
    lpei.update(model=model, eta=1.0)

    configurations = [
        ConfigurationMock([0.1, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    batch_configs = [
        ConfigurationMock([0.6, 0.6, 0.6]),
        ConfigurationMock([0.7, 0.7, 0.7])
    ]

    lpei.batch_configs = batch_configs

    acq = lpei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.25386894076384675)
    assert np.isclose(acq[1][0], 0.21615949314476543)
    assert np.isclose(acq[2][0], -0.09196293793625696)


# --------------------------------------------------------------
# Test PI
# --------------------------------------------------------------

@pytest.fixture
def acq_pi(model):
    return PI(model)

def test_pi_init(acq_pi):
    assert acq_pi.long_name == 'Probability of Improvement'

def test_pi_1xD(model, acq_pi):
    pi = acq_pi
    pi.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = pi(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.7602499389065233)


def test_pi_1xD_zero(model, acq_pi):
    pi = acq_pi
    pi.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([100, 100, 100])]
    acq = pi(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0)


def test_pi_NxD(model, acq_pi):
    pi = acq_pi
    pi.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = pi(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.0)
    assert np.isclose(acq[1][0], 0.99778673707104)
    assert np.isclose(acq[2][0], 0.5)


# --------------------------------------------------------------
# Test LCB
# --------------------------------------------------------------


@pytest.fixture
def acq_lcb(model):
    return LCB(model)

def test_lcb_init(acq_lcb):
    assert acq_lcb.long_name == 'Lower Confidence Bound'
    assert acq_lcb.num_data == None

def test_lcb_1xD(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, eta=1.0, par=1, num_data=3)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 1.315443985917585)
    lcb.update(model=model, eta=1.0, par=1, num_data=100)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 2.7107557771721433)


def test_lcb_1xD_no_improvement_vs_improvement(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, par=1, num_data=1)
    configurations = [ConfigurationMock([100, 100])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], -88.22589977)
    configurations = [ConfigurationMock([0.001, 0.001])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.03623297)


def test_lcb_NxD(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, eta=1.0, num_data=100)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = lcb(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.045306943655446116)
    assert np.isclose(acq[1][0], 1.3358936353814157)
    assert np.isclose(acq[2][0], 3.5406943655446117)

def test_lcb_fail(model, acq_lcb):
    with pytest.raises(ValueError):
        acq_lcb.update(model=model, eta=1.0, par=1)


# --------------------------------------------------------------
# Test Uncertainty
# --------------------------------------------------------------

@pytest.fixture
def acq_uncer(model):
    return Uncertainty(model)

def test_uncer_init(acq_uncer):
    assert acq_uncer.long_name == 'Uncertainty'
    assert acq_uncer.num_data == None

def test_uncertainty_1x1(model, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model, num_data=10)

    # 创建配置实例
    configurations = [ConfigurationMock([0.5])]

    # 使用Uncertainty类计算不确定性值
    acq = uncertainty(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 2.1459660262893476)

def test_uncertainty_NxD(model, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model, num_data=10)

    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]

    # 使用Uncertainty类计算不确定性值
    acq = uncertainty(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.6166458991812724)
    assert np.isclose(acq[1][0], 1.0680620276609596)
    assert np.isclose(acq[2][0], 3.377508689746394)
