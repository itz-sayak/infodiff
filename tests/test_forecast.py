import numpy as np

from infodiff.experiments.track_d import msx_forecast, realised
from infodiff.models.dictionary import PhaseTypeDictionary
from infodiff.models.features import DesignSpec
from infodiff.models.params import model_from_truth
from infodiff.sim.cluster import HawkesTruth, simulate

ENDO = PhaseTypeDictionary.log_grid(0.1, 10.0, 3, orders=2)
EXO = PhaseTypeDictionary.log_grid(1.0, 30.0, 2, orders=3)


def test_closed_form_forecast_is_unbiased():
    d = 2
    A = np.zeros((d, d, ENDO.K, ENDO.R))
    A[0, 0, 1, 0] = 0.4
    A[1, 0, 0, 1] = 0.25
    A[0, 1, 2, 1] = 0.2
    B = np.zeros((d, 1, EXO.K, EXO.R, 2))
    B[0, 0, 0, 2, 0] = 6.0
    B[1, 0, 1, 0, 1] = 3.0
    truth = HawkesTruth(endo=ENDO, A=A, mu=np.array([0.3, 0.2]), exo=EXO, B=B)
    W = 1500
    t0 = np.arange(W) * 1000.0
    rng = np.random.default_rng(1)
    news = [[(t0[w] + 200.0, 0, np.array([1.0, rng.uniform(0, 2)]))] for w in range(W)]
    data = simulate(truth, t0=t0, t1=t0 + 400.0, burn=100.0, news=news, seed=5)
    model = model_from_truth(truth, DesignSpec(endo=ENDO, exo=EXO), data)
    hz = (5.0, 30.0, 150.0)
    pred = np.array([msx_forecast(model, data, w, t0[w] + 200.0, [w], hz) for w in range(W)])
    real = np.array([realised(data, w, t0[w] + 200.0, hz) for w in range(W)])
    m_p, m_r = pred.mean(0), real.mean(0)
    se = real.std(0) / np.sqrt(W)
    assert np.all(np.abs(m_p - m_r) < 4 * se + 0.02), (m_p, m_r, se)
    # conditional informativeness: the history/surprise-conditional forecast beats the
    # unconditional mean in squared error for every horizon and dimension
    mse_cond = ((pred - real) ** 2).mean(0)
    mse_unc = ((m_r[None] - real) ** 2).mean(0)
    assert np.all(mse_cond < mse_unc), (mse_cond, mse_unc)
