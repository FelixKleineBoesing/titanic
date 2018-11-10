import numpy as np


def calculate_class(y_pred: np.ndarray, mode: str):
    assert isinstance(y_pred, np.ndarray), "Prediction has to be list"
    assert mode in ["naive", "optim"], "Mode can only be naive or optim"

    def _determine_class(x: float, prob: float):
        if x > prob:
            return 1
        else:
            return 0

    _determine_class = np.vectorize(_determine_class)

    if mode=="naive":
        y = _determine_class(y_pred, 0.5)
    elif mode=="optim":
        raise AssertionError("Not implemented yet")

    return y

