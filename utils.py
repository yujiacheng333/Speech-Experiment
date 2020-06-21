
from tensorflow.python.keras import backend as k
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
tf.enable_eager_execution()
eps = 1e-8

def trim_silence(audio, rate=.1):
    audio = tf.signal.frame(audio, frame_length=512, frame_step=512)
    power = np.std(audio, axis=1)
    threhold = np.mean(power) * rate
    amp_mask = power < threhold
    right = 0
    left = 0
    for i in range(len(amp_mask)):
        if amp_mask[i]:
            right += 1
        else:
            break

    for i in range(len(amp_mask)):
        if amp_mask[-(i+1)]:
            left += 1
        else:
            break
    if left == 0:
        audio = audio[right:]
    else:
        audio = audio[right:-left]
    return tf.signal.overlap_and_add(audio, frame_step=512)


def pad_audios(audio,  org_length, audio_length, right_pad=False):
    audio = audio[:org_length]
    org_length = org_length
    if org_length > audio_length:
        diff_l = org_length - audio_length
        left = np.random.randint(0, diff_l)
        right = diff_l - left
        audio = audio[left:-right]
        padding = [0, 0]
    elif org_length < audio_length:
        diff_l = np.abs(org_length - audio_length)
        if right_pad:
            left = 0
        else:
            left = np.random.randint(0, diff_l)
        right = diff_l - left
        audio = np.pad(array=audio, pad_width=(left, right))
        padding = [left, right]
    else:
        padding = [0, 0]
    return audio, padding



def normalize(x):

    x -= np.mean(x)
    return x / np.max(np.abs(x))


def whitting(x):
    x = tf.cast(x, tf.float32)
    mean, var = tf.nn.moments(x, keep_dims=True, axes=-1)
    x = (x - mean) / k.sqrt(var + eps)
    return x


def create_mixture(x, snrs):
    aerfas = 10**(snrs/10)
    x = np.asarray(x)
    x *= aerfas[:, np.newaxis]
    return np.sum(x, axis=0)


def mix_audios(audio_array,
               paddings,
               mixnum,
               dataexpand_rate,
               spk_num):
    """

    :param audio_array: [Total, time]
    :param paddings: [Total]
    :param  mix_num: Max source num
    :param dataexpand_rate: not epoch expand
    :return: [Total * dataexpand_rate , time * (maxmix_num + 1)], # mix data first
            ,[Total * dataexpand_rate, max_mix_num]
    """
    total_mix_num = 10 * spk_num * dataexpand_rate
    stepdata_list = []
    padding_list = []
    spker_list = []
    for i in range(total_mix_num):
        spk_index = np.random.choice(spk_num, mixnum, replace=False)
        speech_index = np.random.randint(0, 10, size=mixnum)
        local_fetch = []
        local_padding = []
        for j in range(mixnum):
            local_fetch.append(audio_array[spk_index[j], speech_index[j], :])
            local_padding.append(paddings[spk_index[j], speech_index[j]])
        snr = np.zeros([mixnum])
        snr[1:] = np.random.uniform(-3, 3, size=[mixnum-1])
        mixture = create_mixture(local_fetch, snr)[np.newaxis]
        local_fetch = np.asarray(local_fetch)
        step_data = np.concatenate([mixture, local_fetch], axis=0)
        stepdata_list.append(step_data)
        padding_list.append(np.max(local_padding))
        spker_list.append(spk_index) # delete "#" when need the speaker label
    return np.asarray(stepdata_list).astype(np.float32), np.asarray(padding_list).astype(np.int16),\
           np.asarray(spker_list).astype(np.int16),



tf.enable_eager_execution()


def end_padding(signal, target_length):
    gap = target_length - len(signal)
    signal = np.pad(signal, (0, gap), mode="constant")
    return signal, gap


def makeroom(source, room_sz, absorption, normalize_macs):
    num_source = len(source)
    max_dist = min(room_sz[0], room_sz[1]) / 2.
    room_center = room_sz / 2.
    mac_hight = np.random.uniform(low=0.6, high=1)
    normalize_macs[2, :] += mac_hight
    normalize_macs[:2, :] += room_center[:2, np.newaxis]
    rectfied_macs = pra.MicrophoneArray(normalize_macs, 8000)
    room = pra.ShoeBox(room_sz, absorption=absorption, max_order=3, mics=rectfied_macs)
    spk_dist = np.random.uniform(low=max_dist/2., high=max_dist, size=[num_source])
    shift = np.random.uniform(0, np.pi*2)
    spk_angle = np.random.choice(np.arange(0, np.pi*2, np.pi/3.), num_source) + shift
    spk_x = np.cos(spk_angle) * spk_dist + room_center[0]
    spk_y = np.sin(spk_angle) * spk_dist + room_center[1]
    spk_z = np.random.uniform(1.5, 2.0, size=num_source)
    for i in range(num_source):
        room.add_source(position=[spk_x[i], spk_y[i], spk_z[i]], signal=source[i])
    room.compute_rir()
    snr = np.random.uniform(5, 15)
    room.simulate(snr=snr)
    recv_signals = room.mic_array.signals
    recv_signals = recv_signals.astype(np.float32)

    """
    label_signal
    """
    label_signals = []
    gaps = []
    room_center[-1] = mac_hight
    room_center = room_center[:, np.newaxis]
    room_center_mac = pra.MicrophoneArray(room_center, 8000)
    for i in range(num_source):
        room = pra.ShoeBox(room_sz, absorption=1., max_order=0, mics=room_center_mac)
        room.add_source(position=[spk_x[i], spk_y[i], spk_z[i]], signal=source[i])
        room.compute_rir()
        room.simulate()
        pad_signal, gap = end_padding(room.mic_array.signals[0], target_length=recv_signals.shape[-1])
        label_signals.append(pad_signal)
        gaps.append(gap)
    one_step_data= tf.concat([recv_signals, tf.cast(label_signals, tf.float32)], axis=0)
    print(2)
    return one_step_data, np.max(gaps)


if __name__ == '__main__':
    audios = []
    for i in range(2):
        audios.append(librosa.load("./0_{}.wav".format(i), sr=8000)[0])
    w = np.random.uniform(low=12, high=13)
    l = np.random.uniform(low=12, high=13)
    h = np.random.uniform(low=3, high=4)
    absorption = np.random.uniform(low=0.5, high=1)
    normalize_macs = pra.beamforming.circular_2D_array(center=[0, 0],
                                                       M=6,
                                                       radius=.04,
                                                       phi0=0)
    center_mac = np.asarray([[0], [0]])
    normalize_macs = np.concatenate([normalize_macs, center_mac], axis=1)
    normalize_macs = np.concatenate([normalize_macs, np.zeros([1, 7])], axis=0)
    recv_signals, label_signals, gaps = makeroom(audios, room_sz=np.asarray([w, l, h]), absorption=.9, normalize_macs=normalize_macs)
    """plt.plot(recv_signals[-1])
    plt.show()
    plt.plot(np.sum(label_signals, axis=0))
    plt.show()
    print(np.sum(np.abs(recv_signals[-1] - np.sum(label_signals, axis=0))))
    a = np.abs(tf.signal.stft(recv_signals[-1], frame_length=256, frame_step=128, fft_length=256))
    plt.imshow(a)
    plt.show()
    b = np.abs(tf.signal.stft(np.sum(label_signals, axis=0).astype(np.float32), frame_length=256, frame_step=128, fft_length=256))
    plt.imshow(b)
    plt.show()"""
    process_recv(recv_signals=recv_signals)


