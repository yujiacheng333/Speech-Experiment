import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k

rerank_tabel = np.load("rerank.npy").astype(np.int64)-1
reranktable = rerank_tabel.astype(np.float32)


"""def cutoff_image_randomly(raw, samtic, w, h):
    raw = raw[:w, :h]
    samtic = samtic[:w, :h]
    w, h, c = raw.shape
    diff_w = w - 256
    diff_h = h - 256
    pad_size = []
    if diff_w >= 0:
        shift_w = np.random.randint(0, diff_w, dtype=int)
        raw = raw[shift_w:shift_w + 256]
        samtic = samtic[shift_w:shift_w + 256]
        pad_size.append(0)
    else:
        wpad = -diff_w
        raw = np.pad(raw, ((0, wpad), (0, 0), (0, 0)))
        samtic = np.pad(samtic, ((0, wpad), (0, 0)))
        pad_size.append(wpad)

    if diff_h >= 0:
        shift_h = np.random.randint(0, diff_h, dtype=int)
        raw = raw[:, shift_h:shift_h + 256]
        samtic = samtic[:, shift_h:shift_h + 256]
        pad_size.append(0)
    else:
        hpad = -diff_h
        raw = np.pad(raw, ((0, 0), (0, hpad), (0, 0)))
        samtic = np.pad(samtic, ((0, 0), (0, hpad)))
        pad_size.append(hpad)
    mask = np.zeros([256, 256])
    mask[:256 - pad_size[0], :256 - pad_size[1]] = 1
    return raw, samtic, mask"""


def FLloss(pred_map, cutted_samtic, masks):
    cutted_samtic = tf.one_hot(tf.cast(cutted_samtic, tf.int64), depth=21, dtype=tf.float32)
    masks = tf.cast(masks, tf.float32)
    masks = tf.reshape(masks, [-1, 1])
    pred_map = tf.reshape(pred_map, [-1, 21])
    cutted_samtic = tf.reshape(cutted_samtic, [-1, 21])
    """expterm = tf.exp(pred_map)
    sum_term = k.sum(expterm, axis=-1, keepdims=True)"""
    # FL = (1 - expterm / sum_term) ** 2 * (pred_map - tf.math.log(sum_term)) * cutted_samtic
    FL = (1. - tf.nn.softmax(tf.stop_gradient(pred_map))) ** 2 * tf.nn.log_softmax(pred_map) * cutted_samtic
    return - k.sum(FL * masks) / k.sum(masks)


def embeddingLoss(embedding, cutted_samtic):
    b, w, h, f = embedding.shape
    embedding = tf.reshape(embedding, [b * w * h, f])
    cutted_buffer = cutted_samtic
    cutted_buffer = tf.reshape(cutted_buffer, [b * w * h, -1])
    yyt1 = tf.matmul(cutted_buffer, tf.matmul(cutted_buffer,
                                              np.ones([b * w * h, 1], dtype=np.float32), transpose_a=True))
    yyt1 = 1. / tf.sqrt(yyt1)
    yyt1 = tf.transpose(yyt1)[0, :, tf.newaxis]
    vtv = tf.einsum("bf,bc->fc", embedding * yyt1, embedding)
    yty = tf.einsum("bf,bc->fc", cutted_buffer * yyt1, cutted_buffer)
    vty = tf.einsum("bf,bc->fc", embedding * yyt1, cutted_buffer)
    loss_embedding = (k.sum(vtv ** 2) - 2 * k.sum(vty ** 2) + k.sum(yty ** 2)) / (w * h * f / 10.)
    return loss_embedding


def Diceloss(pred_map, cutted_samtic):
    pred_map = tf.nn.softmax(pred_map)
    cutted_samtic = tf.one_hot(tf.cast(cutted_samtic, tf.int64), depth=21, dtype=tf.float32)
    exsist_weight = tf.clip_by_value(k.sum(cutted_samtic, axis=[1, 2]), clip_value_min=0., clip_value_max=1.).numpy()
    exsist_weight[np.sum(exsist_weight, axis=1) == 1] = 0.
    inter = k.sum(pred_map * cutted_samtic, axis=[1, 2])
    intra = k.sum(pred_map ** 2, axis=[1, 2]) + k.sum(cutted_samtic ** 2, axis=[1, 2])
    loss = (1 - (2 * inter + 1) / (intra + 1)) * exsist_weight
    weight = np.asarray([[1.74518097, 39.60527565, 39.0037343, 36.00139185, 43.58438583, 28.0961859,
                         31.59653758, 23.9822721, 27.83823349, 33.18632839, 33.54881958, 37.16966452,
                         38.40291471, 36.36421256, 36.85893006, 30.02681763, 29.44990249, 35.71968487,
                         33.72870758, 31.0863762, 15.53419507]])
    loss *= weight
    return k.mean(loss)


def Softmaxloss(pred_map, cutted_samtic):
    pred_map = tf.reshape(pred_map, [-1, 21])
    cutted_samtic = tf.cast(cutted_samtic, tf.int64)
    cutted_samtic = tf.reshape(cutted_samtic, [-1])
    celoss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_map, labels=cutted_samtic)
    celoss = celoss
    return k.mean(celoss)


def IOUmetric(x, y, grd_mask):
    y = tf.one_hot(tf.cast(y[..., 0], tf.int64), depth=21, dtype=tf.float32)
    hasobj = k.sum(y, axis=[0, 1, 2]).numpy()
    hasobj = (hasobj !=0)
    x = tf.nn.softmax(x, axis=-1)
    x *= grd_mask
    y *= grd_mask
    inter = np.sum(np.sum(np.logical_and(x, y), axis=1), axis=1)
    union = np.sum(np.sum(np.logical_or(x, y), axis=1), axis=1)
    inter = inter[:, hasobj]
    union = union[:, hasobj]
    iou = inter / union
    return iou

def bce(preds, trues, grd_mask, beta=1, channel_axis=-1):
    preds = tf.nn.softmax(preds, axis=-1)
    trues = tf.one_hot(tf.cast(trues[..., 0], tf.int64), depth=21, dtype=tf.float32)
    loss = tf.keras.losses.BinaryCrossentropy(preds, trues)
    print(1)


def fb_loss(preds, trues, beta=1, channel_axis=-1):
    preds = tf.nn.softmax(preds, axis=-1)
    trues = tf.one_hot(tf.cast(trues, tf.int64), depth=21, dtype=tf.float32)
    smooth = 1e-4
    beta2 = beta * beta
    batch = preds.shape[0]
    classes = preds.shape[channel_axis]
    preds = tf.reshape(preds, [batch, -1, classes])
    trues = tf.reshape(trues, [batch, -1, classes])
    weights = tf.clip_by_value(tf.reduce_sum(trues, axis=1), clip_value_min=0., clip_value_max=1.)
    TP_raw = preds * trues
    TP = tf.reduce_sum(TP_raw, axis=1)
    FP_raw = preds * (1 - trues)
    FP = tf.reduce_sum(FP_raw, axis=1)
    FN_raw = (1 - preds) * trues
    FN = tf.reduce_sum(FN_raw, axis=1)
    Fb = ((1 + beta2) * TP + smooth) / ((1 + beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = tf.reduce_sum(Fb) / (tf.reduce_sum(weights) + smooth)
    return 1. - tf.clip_by_value(score, 0., 1.)
