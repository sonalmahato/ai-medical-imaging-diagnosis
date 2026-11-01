import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("src", "eval"))
from calibrate_and_subgroups import fit_temperature, macro_auroc, sigmoid


def test_temperature_recovers_known_overconfidence():
    rng = np.random.RandomState(0)
    z = rng.normal(0, 1.5, size=(20000, 3))
    y = (rng.uniform(size=z.shape) < sigmoid(z)).astype("float32")   # calibrated at T=1
    t = fit_temperature(z * 2.0, y)                                    # model made 2x overconfident
    assert 1.8 < t < 2.2


def test_temperature_does_not_change_auroc():
    rng = np.random.RandomState(1)
    z = rng.normal(0, 2, size=(5000, 3))
    y = (rng.uniform(size=z.shape) < sigmoid(z)).astype("float32")
    a = macro_auroc(y, sigmoid(z), [0, 1, 2])
    b = macro_auroc(y, sigmoid(z / 1.7), [0, 1, 2])
    assert abs(a - b) < 1e-9
