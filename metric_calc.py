
from evaluation.dataset.util.alignment_gpu import align_depth_least_square
import numpy as np

import torch.nn.functional as F
import torch

def compute_errors(flow_gt, flow_preds, valid_arr):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    a1_arr = []
    a2_arr = []
    a3_arr = []
    abs_rel_arr = []
    rmse_arr = []
    # log_10_arr = []
    # rmse_log_arr = []
    # silog_arr = []
    sq_rel_arr = []

    min_depth_eval = 0.0001
    max_depth_eval = 1

    for gt, pred, valid in zip(flow_gt, flow_preds, valid_arr):

        disparity_pred, scale, shift = align_depth_least_square(
                        gt_arr=gt,
                        pred_arr=pred,
                        valid_mask_arr=valid,
                        return_scale_shift=True,
                    )
        gt = gt.squeeze().cpu().numpy()
        pred = disparity_pred.clone().squeeze().cpu().detach().numpy()
        valid = valid.squeeze().cpu()          
        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval
        pred[np.isnan(pred)] = min_depth_eval

        # pred[pred < min_depth_eval] = min_depth_eval
        # pred[pred > max_depth_eval] = max_depth_eval
        # pred[np.isinf(pred)] = max_depth_eval
        # pred[np.isnan(pred)] = min_depth_eval
        # gt[gt < min_depth_eval] = min_depth_eval
        # gt[gt > max_depth_eval] = max_depth_eval
        # gt[np.isinf(gt)] = max_depth_eval
        # gt[np.isnan(gt)] = min_depth_eval

        gt, pred= gt[valid.bool()], pred[valid.bool()]

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = (np.abs(gt - pred) / gt).mean()
        sq_rel =(((gt - pred) ** 2) / gt).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(gt) - np.log(pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        # err = np.log(pred) - np.log(gt)
        # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        # log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        a1_arr.append(a1)
        a2_arr.append(a2)
        a3_arr.append(a3)
        abs_rel_arr.append(abs_rel)
        rmse_arr.append(rmse)
        # log_10_arr.append(log_10)
        # rmse_log_arr.append(rmse_log)
        # silog_arr.append(silog)
        sq_rel_arr.append(sq_rel)

    a1_arr_mean = sum(a1_arr) / len(a1_arr)
    a2_arr_mean = sum(a2_arr) / len(a2_arr)
    a3_arr_mean = sum(a3_arr) / len(a3_arr)
    abs_rel_arr_mean = sum(abs_rel_arr) / len(abs_rel_arr)
    rmse_arr_mean = sum(rmse_arr) / len(rmse_arr)
    # log_10_arr_mean = sum(log_10_arr) / len(log_10_arr)
    # rmse_log_arr_mean = sum(rmse_log_arr) / len(rmse_log_arr)
    # silog_arr_mean = sum(silog_arr) / len(silog_arr)
    sq_rel_arr_mean = sum(sq_rel_arr) / len(sq_rel_arr)

    # return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, log_10=log_10_arr_mean, rmse_log=rmse_log_arr_mean,
    #             silog=silog_arr_mean, sq_rel=sq_rel_arr_mean)
    return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, sq_rel=sq_rel_arr_mean)

def sequence_loss(flow_preds, flow_gt, valid, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # L1 loss
    flow_loss = F.l1_loss(flow_preds[valid.bool()], flow_gt[valid.bool()])
    metrics = compute_errors(flow_gt, flow_preds, valid)

    
    return flow_loss, metrics