import numpy as np
from mir_eval.separation import bss_eval_sources


def cal_sdri(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_sdri = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    return avg_sdri


def cal_sisnri(src_ref, src_est, mix):
    """

    :param src_ref: [spk, T]
    :param src_est: [spk, T]
    :param mix:
    :return:
    """
    sisnr1 = cal_sisnr(src_ref[0], src_est[0])
    sisnr2 = cal_sisnr(src_ref[1], src_est[1])
    sisnr1b = cal_sisnr(src_ref[0], mix)
    sisnr2b = cal_sisnr(src_ref[1], mix)
    avg_sisnri = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_sisnri


def cal_sisnr(ref_sig, out_sig, eps=1e-8):
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr
