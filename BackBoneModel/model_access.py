import os
import numpy as np
import tensorflow as tf
from utils import pad_audios
from scipy.io import wavfile
from Dataloaders import SourceSeparationSet
from BackBoneModel.DPRNN import DPCLnet
from tensorflow.python.keras import backend as k
from sklearn.cluster import KMeans

tf.enable_eager_execution()


class ModelAccess(object):
    def __init__(self, action, TIMIT_dir, ckpt_dir):
        super(ModelAccess, self).__init__()
        self.kmeans = KMeans(n_clusters=2)
        assert action in ["train", "test", "import"], "Not in action set"
        record_loader = SourceSeparationSet.Sepdataset(source_folder=TIMIT_dir)
        self.train_dataset = record_loader.traindataset
        self.eval_dataset = record_loader.evaldataset
        self.test_dataset = record_loader.testdataset
        self.audio_length = 16000
        self.num_mixture_spk = 2
        # batch wise expand =>epochs * 2
        self.epochs = 600
        self.lr = tf.Variable(0.)
        self.max_lr = 1e-2
        # OOM AT bs 9, K=23, STRIDE=5
        self.batch_size = 64
        self.warm_up_step = 500
        self.decay_step = 100
        self.decay_rate = 0.98
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.max_to_keep = 5
        self.num_spks = int(record_loader.M - record_loader.T0)
        self.B = DPCLnet()
        self.current_step = tf.Variable(0)
        self.current_epoch = tf.Variable(0)
        self.optimizer = tf.train.AdamOptimizer(5e-4)
        self.checkpoint = tf.train.Checkpoint(epoch=self.current_epoch,
                                              step=self.current_step,
                                              optimizer=self.optimizer,
                                              model=self.B)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=self.max_to_keep)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)

        if action == "train":

            while self.current_epoch.numpy() < self.epochs:
                self.work_on_dataset(self.train_dataset, training=True)
                self.ckpt_manager.save()
                self.work_on_dataset(self.test_dataset, training=False)
                self.current_epoch.assign_add(1)
            self.work_on_dataset(self.test_dataset, training=False)
        else:
            print("finish load model!")

    def work_on_dataset(self, dataset, training):

        if training:
            dataset = dataset.shuffle(5040, reshuffle_each_iteration=True).repeat(2)
        else:
            dataset = dataset.shuffle(6300-5040, reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(self.batch_size*2,
                                       padded_shapes=(tf.TensorShape([None]),
                                                      tf.TensorShape([]),
                                                      tf.TensorShape([])),
                                       drop_remainder=True)
        local_iter = dataset.make_one_shot_iterator()

        for batch_data in local_iter:

            speech_0, speech_1 = np.split(batch_data[0].numpy(), 2, axis=0)
            spk_0, spk_1 = np.split(batch_data[1].numpy(), 2, axis=0)
            length_0, length_1 = np.split(batch_data[2].numpy(), 2, axis=0)
            reptimes = np.sum(spk_0 == spk_1)
            if reptimes > self.batch_size:
                continue
            for i in range(10):
                reshuffle_index = np.random.choice(np.arange(self.batch_size), replace=False, size=self.batch_size)
                spk_1 = spk_1[reshuffle_index]
                reptimes = np.sum(spk_0 == spk_1)
                if reptimes == 0:
                    speech_1 = speech_1[reshuffle_index]
                    length_1 = length_1[reshuffle_index]
            speechs = np.concatenate([speech_0, speech_1], axis=0)
            lengths = np.concatenate([length_0, length_1], axis=0)
            pad_list = []
            speech_list = []
            for local_speech, local_length in zip(speechs, lengths):
                s, p = pad_audios(local_speech, org_length=local_length, audio_length=self.audio_length)
                pad_list.append(p)
                speech_list.append(s)
            speech_list = tf.cast(speech_list, tf.float32)
            mean, var = tf.nn.moments(speech_list, axes=[1], keep_dims=True)
            speech_list -= mean
            speech_list /= tf.sqrt(var)
            speech_list = speech_list.numpy()
            speech_0, speech_1 = np.split(speech_list, 2, axis=0)
            mixture = speech_0 + speech_1
            one_batch_data = tf.concat([mixture, speech_0, speech_1], axis=0)
            one_batch_data = tf.signal.stft(one_batch_data, frame_length=256, frame_step=64, fft_length=256)
            angle = np.angle(one_batch_data[0])
            one_batch_data = tf.abs(one_batch_data)
            one_batch_data /= k.max(one_batch_data, axis=[1, 2], keepdims=True)
            mixture, speech_0, speech_1 = np.split(one_batch_data.numpy(), 3, axis=0)
            valid_data = (mixture > 1e-7).astype(np.float32).reshape([self.batch_size, -1, 1])
            ibm = tf.one_hot((speech_0 > speech_1).astype(np.int64), depth=2)
            y = tf.reshape(ibm, [self.batch_size, -1, 2])
            with tf.GradientTape() as tape:
                est_emb = self.B(tf.log1p(mixture))
                v = tf.reshape(est_emb, [self.batch_size, -1, 40])
                if training:
                    v *= valid_data
                    y *= valid_data
                    vt = tf.transpose(v, [0, 2, 1])
                    yt = tf.transpose(y, [0, 2, 1])
                    vtv = k.sum(tf.matmul(vt, v)**2, axis=[1, 2])
                    yty = k.sum(tf.matmul(yt, y)**2, axis=[1, 2])
                    vty = k.sum(tf.matmul(vt, y)**2, axis=[1, 2])
                    loss = (vtv+yty-2*vty) / k.sum(valid_data)
                    loss = k.sum(loss) / 1000.
                    grd = tape.gradient(loss, self.B.trainable_variables)
                    print(loss)
                    self.optimizer.apply_gradients(zip(grd, self.B.trainable_variables))
                else:
                    est = self.kmeans.fit_predict(v.numpy()[0])
                    est = est.reshape([247, 129]).astype(np.int64)
                    est = tf.one_hot(est, depth=2).numpy().transpose([2, 0, 1])
                    mixture = mixture[0].reshape([1, 247, 129]).astype(np.complex64)

                    mixture *= np.exp(1j*angle[np.newaxis])
                    est_speech = tf.signal.inverse_stft(est * mixture, frame_length=256, frame_step=64, fft_length=256)
                    mixture = tf.signal.inverse_stft(mixture, frame_length=256, frame_step=64, fft_length=256)
                    wavfile.write("mixture.wav", 8000, mixture[0].numpy())
                    wavfile.write("recon_0.wav", 8000, est_speech[0].numpy()/np.max(np.abs(est_speech[0].numpy())))
                    wavfile.write("recon_1.wav", 8000, est_speech[1].numpy()/np.max(np.abs(est_speech[0].numpy())))
                    print(1)
                    break

    def warm_up_dacay_lr(self):
        if self.current_step < self.warm_up_step:
            lr = self.current_step.numpy() / self.warm_up_step * (self.max_lr - self.min_lr) + self.min_lr
        else:
            lr = self.max_lr * self.decay_rate ** ((self.current_step.numpy() - self.warm_up_step) / self.decay_step)
        self.lr.assign(lr)


if __name__ == '__main__':
    ModelAccess("train", TIMIT_dir="D:/code/Ada-EA-S/TIMIT_data",
                ckpt_dir="D:/code/Ada-EA-S/Ckpts/backbone")

