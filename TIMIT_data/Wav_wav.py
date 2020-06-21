import os
from sphfile import SPHFile


def _getall_spk(targetfp):
    spk_list = []
    for i in os.listdir(targetfp):
        for j in os.listdir(targetfp+"/"+i):
            spk_list.append(j)
    return spk_list


def transform_spkinfo(targetfp, spk_list):
    for i in os.listdir(targetfp):
        local_fp = targetfp + "/" + i
        for j in os.listdir(local_fp):
            counter = 0
            spk_index = spk_list.index(j)
            print(spk_index)
            audios = os.listdir(targetfp + "/" + i + "/" + j)
            for k in audios:
                if ".WAV" in k:
                    subfp = targetfp + "/" + i + "/" + j + "/" + k

                    sph = SPHFile(subfp)
                    sph.write_wav(filename="./all_wav/" +
                                           str(spk_index) + "_"+str(counter) + ".wav")
                    counter += 1
            if counter != 10:
                raise ValueError("Not enough speech")


def Runonfloader(target_fp, fp="F:\TIMIT\concat"):

    os.makedirs(target_fp+"/all_wav", exist_ok=True)

    spk_list = _getall_spk(fp)

    spk_str = ""

    for i in spk_list:
        spk_str += (i + "_")

    with open(target_fp+"/spklist.txt", "w") as f:
        f.writelines(spk_str)

    transform_spkinfo(targetfp=fp, spk_list=spk_list)

if __name__ == '__main__':
    Runonfloader(target_fp="D:/code/Ada-EA-S/TIMIT_data")