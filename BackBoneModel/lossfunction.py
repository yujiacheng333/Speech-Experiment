import tensorflow as tf
import numpy as np
from itertools import permutations
from tensorflow.python.keras import backend as k
eps = 1e-8


def reorder_source(source, perms, max_snr_idx):
    """
    reoder source by max_snr_idx, onehot (bs, rank)->(bs, rank_, onehot) matmul (bs, source, time)
    rerank source
    :param source: to rerank [bs, spk, time]
    :param perms: perms [spk!, spk, spk onthot]
    :param max_snr_idx: [bs, spk!] (onehot from [bs])
    :return:
    """
    perms = tf.cast(perms, tf.float32)
    max_snr_idx = tf.cast(max_snr_idx, tf.float32)
    local_rank = tf.einsum("pj,jkf->pkf", max_snr_idx, perms)
    reoderde_source = tf.matmul(local_rank, source)
    return reoderde_source


def cal_si_snr_with_pit(source, estimate_source, padding):
    """
    :param source:[bs, spkt, time]
    :param estimate_source: [B, spke, time]
    :param source_lengths: source_length to remove pad
    :return:
    """
    assert source.shape == estimate_source.shape
    bs, spk1, time = source.shape
    mask = get_mask(source, padding)
    source *= mask
    estimate_source *= mask
    mean_target = k.mean(source, axis=[2], keepdims=True)
    mean_estimate = k.mean(estimate_source, axis=[2], keepdims=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask  # [bs, spkt, time]
    zero_mean_estimate *= mask  # [bs, spke, time]
    pair_wise_dot = tf.matmul(zero_mean_estimate, zero_mean_target, transpose_b=True)
    # [bs, spkt, spke]
    s_target_energy = k.sum(zero_mean_target**2, axis=-1, keepdims=True) + eps
    s_target_energy = s_target_energy[:, tf.newaxis, :, :]
    # [bs, spkt ,1]
    # s_target_energy = k.sum(s_target**2, axis=-1, keepdims=True) + eps
    # [bs, spkt, spke, 1]
    # s_target = <s', s>s / ||s||^2
    # s' []
    pair_wise_proj = pair_wise_dot[..., tf.newaxis] * zero_mean_target[:, tf.newaxis, :, :] / s_target_energy
    e_noise = zero_mean_estimate[:, :, tf.newaxis, :] - pair_wise_proj
    pair_wise_si_snr = k.sum(pair_wise_proj ** 2, axis=-1)/(k.sum(e_noise**2, axis=-1)+eps)
    pair_wise_si_snr = 10 * tf.math.log(pair_wise_si_snr+eps) / tf.math.log(10.)
    # permutations, [C!, C]
    perms = tf.cast(list(permutations(range(spk1))), tf.int64)
    length = perms.shape[0]
    perms = tf.one_hot(perms, depth=spk1)
    # perms [C!, C , C]
    snr_set = tf.einsum("bij,pij->bp", pair_wise_si_snr, perms)
    max_snr_idx = k.argmax(snr_set, axis=-1)  # [B,]
    max_snr_idx = tf.one_hot(max_snr_idx, depth=length)
    max_snr = max_snr_idx * snr_set
    max_snr = k.sum(max_snr, axis=-1) / tf.cast(spk1, tf.float32)
    return max_snr, perms, max_snr_idx


def get_mask(source, padlist):
    """

    :param source: [bs, spk1, time]
    :param time: [bs, ]
    :return: [bs, spk1, time] in 0, 1 as mask
    """
    mask_out = np.ones_like(source)
    for i, l in enumerate(padlist):
        mask_out[i, :, :l[0]] = 0
        mask_out[i, :, 16384-l[1]:] = 0
    return mask_out


def cal_loss(source, estimate_source, padding):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      padding)
    loss = - k.mean(max_snr)
    # reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss  # , max_snr, estimate_source, reorder_estimate_source


