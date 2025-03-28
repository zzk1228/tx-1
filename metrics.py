import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

# 标准差
def STD(pred, true):
    return np.std(pred - true)

def metric(pred, true, batch_size=1000):
    """Calculate metrics in batches to avoid memory issues"""
    if isinstance(pred, np.ndarray) and isinstance(true, np.ndarray):
        total_samples = pred.shape[0]
        if total_samples > batch_size:
            num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
            
            mae_total, mse_total, rmse_total, mape_total, mspe_total = 0, 0, 0, 0, 0
            sample_count = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                current_batch_size = end_idx - start_idx
                
                batch_pred = pred[start_idx:end_idx]
                batch_true = true[start_idx:end_idx]
                
                # Calculate metrics for this batch
                batch_mae = MAE(batch_pred, batch_true)
                batch_mse = MSE(batch_pred, batch_true)
                batch_rmse = RMSE(batch_pred, batch_true)
                batch_mape = MAPE(batch_pred, batch_true)
                batch_mspe = MSPE(batch_pred, batch_true)
                
                # Weighted average based on batch size
                sample_count += current_batch_size
                mae_total += batch_mae * current_batch_size
                mse_total += batch_mse * current_batch_size
                rmse_total += batch_rmse * current_batch_size
                mape_total += batch_mape * current_batch_size
                mspe_total += batch_mspe * current_batch_size
            
            return mae_total/sample_count, mse_total/sample_count, rmse_total/sample_count, mape_total/sample_count, mspe_total/sample_count
        else:
            # If data is small enough, use the original calculation
            mae = MAE(pred, true)
            mse = MSE(pred, true)
            rmse = RMSE(pred, true)
            mape = MAPE(pred, true)
            mspe = MSPE(pred, true)
            return mae, mse, rmse, mape, mspe
    else:
        # Handle case when inputs are not numpy arrays
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)
        return mae, mse, rmse, mape, mspe