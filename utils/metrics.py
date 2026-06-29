import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def _select_masked_values(pred, true, mask):
    pred = np.asarray(pred)
    true = np.asarray(true)
    mask = np.asarray(mask)

    try:
        broadcast_mask = np.broadcast_to(mask.astype(bool), pred.shape)
    except ValueError as exc:
        raise ValueError(
            f"Mask with shape {mask.shape} cannot broadcast to prediction shape {pred.shape}."
        ) from exc

    if not np.any(broadcast_mask):
        raise ValueError("Mask has no valid evaluation points.")

    return pred[broadcast_mask], true[broadcast_mask]


def metric(pred, true, mask=None):
    if mask is None:
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)

        return mae, mse, rmse, mape, mspe

    pred, true = _select_masked_values(pred, true, mask)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
