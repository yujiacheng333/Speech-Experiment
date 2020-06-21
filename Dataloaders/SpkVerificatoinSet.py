import tensorflow as tf
import numpy as np
import librosa
from utils import pad_audios, trim_silence
tf.enable_eager_execution()


class SpkDataset(object):
    def __init__(self, source_folder, reset=False):
        """
         TIMIT: spk:M Speech:N

 Classifier: spk: M-T0, Speech:N
             train:  spk:M-T0, Speech:N-T1
             test:   spk:M-T0, Speech:T1
 SourceSeprate: Spk: M, Speech N
                train: spk:M-T0-T2, Speech:N
                eval:  spk:T2,      Speech:N
                test:  spk:T0,      Speech:N
       0.7 train, 0.2 eval, 0.1 test
        same expand rate->target hour is 30Hour train:
        data expand = 30 * 3600 / 16384 * 8000 / 630 / 10
        our setting ： dataexpand = 4.
        """
        self.M = 630
        self.N = 10
        self.T0 = 63
        self.T1 = 2
        self.reset = reset
        with open(source_folder + "/spklist.txt", "r") as f:
            self.spk_list = f.read()
        self.spk_list = self.spk_list.split("_")[:-1]
        self.record_prefix = "Verify"
        self.fs = 8000
        self.audio_length = 16384

        if self.reset:
            # self.audio_array_train = np.zeros([self.M - self.T0, self.N - self.T1, self.audio_length*2])
            # self.audio_array_test = np.zeros([self.M - self.T0, self.T1, self.audio_length*2])
            writer_train = tf.io.TFRecordWriter(source_folder + "/" + self.record_prefix + "_train.tfrecord")
            writer_test = tf.io.TFRecordWriter(source_folder + "/" + self.record_prefix + "_test.tfrecord")
            counter = 0
            for spk in range(self.M - self.T0):
                for rank in range(self.N):
                    local_fp = source_folder + "/all_wav/{}_{}.wav".format(spk, rank)
                    raw_audio = librosa.load(local_fp, sr=self.fs, mono=True)[0]
                    raw_audio -= np.mean(raw_audio)
                    raw_audio = trim_silence(raw_audio, rate=.1)
                    raw_audio /= np.max(raw_audio)
                    if rank < self.N - self.T1:
                        writer_train.write(self._serialize_example(raw_audio, spk))
                    else:
                        writer_test.write(self._serialize_example(raw_audio, spk))
                    counter += 1
                print(counter)
            writer_train.close()
            writer_test.close()

        raw_dataset_train = tf.data.TFRecordDataset(source_folder + "/" + self.record_prefix + "_train.tfrecord")
        self.traindataset = raw_dataset_train.map(self._extract_fn)

        raw_dataset_test = tf.data.TFRecordDataset(source_folder + "/" + self.record_prefix + "_test.tfrecord")
        self.testdataset = raw_dataset_test.map(self._extract_fn)
        del raw_dataset_train, raw_dataset_test


    def _extract_fn(self, data_record):
        features = {
            'speech': tf.io.VarLenFeature(tf.string),
            'spk_label': tf.io.FixedLenFeature([], tf.int64),
            'length': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(tf.sparse.to_dense(sample["speech"], default_value=b''), tf.float32)
        spk_label = sample["spk_label"]
        length = sample["length"]
        return tf.squeeze(recv_sig, axis=0), spk_label, length


    def _serialize_example(self, speech, spk_label):
        l_speech = len(speech)
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        signals = speech.numpy().astype(np.float32)
        feature = {
            'speech': _bytes_feature(signals.tostring()),
            'spk_label': _int64_feature(np.int64(spk_label)),
            'length': _int64_feature(np.int64(l_speech))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


if __name__ == '__main__':
    a = SpkDataset(source_folder="D:/code/Ada-EA-S/TIMIT_data", reset=True)
    for i in a.traindataset:
        speech = i[0]
        length = i[2]
        padded_speech = pad_audios(speech, org_length=length, audio_length=a.audio_length)[0]
        print(padded_speech.shape)