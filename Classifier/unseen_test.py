import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k
from Classifier.classfier_access import ModelAccess
from utils import pad_audios
import librosa

tf.enable_eager_execution()

if __name__ == '__main__':
    model = ModelAccess("import", TIMIT_dir="D:\YuJiacheng333codeprefix\codes\Ada-U-net-EA-S\TIMIT_data",
                ckpt_dir="D:\YuJiacheng333codeprefix\codes\Ada-U-net-EA-S\Ckpts").V
    # generate spk id
    audio_set = []
    for i in range(63):
        local_name = "{}_0.wav".format(629-i)
        local_audio = librosa.load("../TIMIT_data/all_wav/"+local_name, sr=8000)[0]
        audio_set.append(pad_audios(local_audio, org_length=int(len(local_audio)), audio_length=16384)[0])
    id_set= model(inputs=tf.cast(audio_set, tf.float32)
                  , training=False, return_embedding=True)
    id_set = tf.nn.l2_normalize(k.sum(id_set, axis=[1]), axis=-1)
    inter_proj = tf.einsum("if,jf->ij", id_set, id_set)
    import matplotlib.pyplot as plt
    plt.imshow(inter_proj.numpy())
    plt.show()
    audio_set = np.zeros([63, 10, 16384])
    audio_label = np.zeros([63, 10])
    for i in range(63):
        audio_label[i, :] = i
        for j in range(10):
            local_name = "{}_{}.wav".format(629-i, j)
            local_audio = librosa.load("../TIMIT_data/all_wav/"+local_name, sr=8000)[0]
            audio = (pad_audios(local_audio, org_length=int(len(local_audio)), audio_length=16384)[0]).numpy()
            audio_set[i, j, :] += audio
    audio_set = audio_set.reshape([-1, 16384])
    query_set= model(inputs=tf.cast(audio_set, tf.float32), training=False, return_embedding=True)
    query_set = tf.nn.l2_normalize(k.sum(query_set, axis=[1]), axis=-1)
    projection = tf.einsum("bf,qf->bq", query_set, id_set)
    print(np.mean(k.argmax(projection, axis=-1).numpy()==audio_label.reshape([-1])))
    mat = tf.confusion_matrix(labels=audio_label.reshape([-1]), predictions=k.argmax(projection, axis=-1), num_classes=63)
    import matplotlib.pyplot as plt
    plt.imshow(mat.numpy())
    plt.show()
    projection = tf.nn.softmax(tf.reshape(projection, [63, 10, 63]), axis=-1).numpy()
    label = tf.one_hot(tf.cast(audio_label, tf.int64), depth=63).numpy()
    threhold_line = np.arange(0, 1, 0.01)[np.newaxis]
    TPs = []
    FPs = []
    for i in range(63):
        T_logits = projection[i]
        F_logits = np.delete(projection, i, 0).reshape([-1, 63])
        T_logits = T_logits[:, i][..., np.newaxis]
        F_logits = F_logits[:, i][..., np.newaxis]
        TP = np.sum((T_logits > threhold_line).astype(np.int), axis=0) / 10
        FP = np.sum((F_logits > threhold_line).astype(np.int), axis=0) / (62 * 10)
        TPs.append(TP)
        FPs.append(FP)
    TPs = np.mean(TPs, axis=0)
    FPs = np.mean(FPs, axis=0)
    plt.plot(FPs, TPs)
    plt.show()
    print(FPs[np.argmin(np.abs(1 - FPs - TPs))])