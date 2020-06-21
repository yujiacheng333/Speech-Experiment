import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.applications import ResNet50
# from Classifier.Non_uttrance_classifier import NUClassifier
# from Classifier.Deep_speaker import Resnet18
from Classifier.classifier import Embedding_network
# from Classifier.mobilenetV3 import MobilenetV3
from Dataloaders import SpkVerificatoinSet
from utils import pad_audios

tf.enable_eager_execution()


class ModelAccess(object):
    def __init__(self, action, TIMIT_dir, ckpt_dir):
        super(ModelAccess, self).__init__()
        assert action in ["train", "test", "import", "get_embedding"], "Not in action set"
        record_loader = SpkVerificatoinSet.SpkDataset(source_folder=TIMIT_dir)
        self.train_dataset = record_loader.traindataset
        self.test_dataset = record_loader.testdataset
        self.epochs = 100
        self.lr = tf.Variable(0.)
        self.max_lr = 1e-2
        self.min_lr = 1e-4
        self.batch_size = 64
        self.warm_up_step = 500
        self.decay_step = 100
        self.decay_rate = 0.98
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.max_to_keep = 5
        # self.V = Verification(int(record_loader.M - record_loader.T0))
        self.num_spks = int(record_loader.M - record_loader.T0)
        self.V = Embedding_network(num_spks=self.num_spks)
        self.current_step = tf.Variable(0)
        self.current_epoch = tf.Variable(0)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.checkpoint = tf.train.Checkpoint(epoch=self.current_epoch,
                                              step=self.current_step,
                                              optimizer=self.optimizer,
                                              model=self.V)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=self.max_to_keep)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)

        if action == "train":

            while self.current_epoch.numpy() < self.epochs:
                self.train_epochs()
                self.ckpt_manager.save()
                self.test_acc()
                self.current_epoch.assign_add(1)

        elif action == "test":
            self.test_acc()

        elif action == "get_embedding":
            self.get_spk_vectors()
        else:
            print("finish load model!")

    def accumulate_iter(self, local_iter):
        batch_speech = []
        batch_label = []
        for step, datas in enumerate(local_iter):
            if step % self.batch_size == 0 and step != 0:
                yield tf.cast(batch_speech, tf.float32),  tf.cast(batch_label, tf.int64)
                batch_speech = []
                batch_label = []
            snr = np.random.uniform(-3, 3)
            ratio = 10.**(snr / 10.)
            noise = np.random.normal(0, ratio)
            batch_speech.append(pad_audios(datas[0], datas[2].numpy(), 16384)[0])
            batch_label.append(datas[1])

    def train_epochs(self):
        local_iter = self.train_dataset.shuffle(10000, reshuffle_each_iteration=True)
        batched_iter = self.accumulate_iter(local_iter)
        for batch_data in batched_iter:
            speech, spk_label = batch_data
            with tf.GradientTape() as tape:
                logits = self.V([speech, spk_label], training=True)
                loss = k.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=spk_label))
                grd = tape.gradient(loss, self.V.trainable_variables)
                self.warm_up_dacay_lr()
                self.optimizer.apply_gradients(grads_and_vars=zip(grd, self.V.trainable_variables))
                self.checkpoint.step.assign_add(1)

    def test_acc(self):
        local_iter = self.test_dataset.shuffle(6000)
        batched_iter = self.accumulate_iter(local_iter)
        acc_list = []
        for batch_data in batched_iter:
            speech, spk_label = batch_data
            logits = self.V(speech, training=False)
            acc_list.append(tf.keras.metrics.sparse_categorical_accuracy(y_pred=logits,y_true= spk_label))
        print("After {} epoch training, Acc on test is {:1.3f}".format(self.current_epoch.numpy(), np.mean(acc_list)))

    def warm_up_dacay_lr(self):
        if self.current_step < self.warm_up_step:
            lr = self.current_step.numpy() / self.warm_up_step * (self.max_lr - self.min_lr) + self.min_lr
        else:
            lr = self.max_lr * self.decay_rate ** ((self.current_step.numpy() - self.warm_up_step) / self.decay_step)
        self.lr.assign(lr)

    def get_spk_vectors(self):
        speech_counter = np.zeros([630-63])
        output = np.zeros([630-63, 10, 256])
        for one_data in self.train_dataset:
            speech, label, length = one_data
            speech = speech[:length]
            embedding = self.V(speech[tf.newaxis, :], return_embedding=True, training=False)
            output[int(label), int(speech_counter[label])] += embedding.numpy()[0]
            speech_counter[label] += 1
        for one_data in self.test_dataset:
            speech, label, length = one_data
            speech = speech[:length]
            embedding = self.V(speech[tf.newaxis, :], return_embedding=True, training=False)
            output[int(label), int(speech_counter[label])] +=embedding.numpy()[0]
            speech_counter[label] += 1
        assert np.sum(speech_counter) == (630-63)*10
        np.save("./speaker_embeddings.npy", output)


if __name__ == '__main__':
    ModelAccess("get_embedding", TIMIT_dir="D:/code/Ada-EA-S/TIMIT_data",
                ckpt_dir="D:/code/Ada-EA-S/Ckpts/classifier")
