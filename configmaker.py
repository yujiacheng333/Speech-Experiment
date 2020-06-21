import json
import os
json_path = "../configs"


class ConfigTables(object):
    """
    This class is used to json file IO op
    """
    def __init__(self):
        self.json_dir = json_path
        self.io_config_fp = self.json_dir + "/" + "ioconfig.json"
        self.data_config_fp = self.json_dir + "/" + "data_config.json"
        self.train_config_fp = self.json_dir + "/" + "train_config.json"
        self.model_config_fp = self.json_dir + "/" + "model_config.json"
        with open(self.io_config_fp, "r") as f:
            self.io_config = json.load(f)
        with open(self.data_config_fp, "r") as f:
            self.data_config = json.load(f)
        with open(self.train_config_fp, "r") as f:
            self.train_config = json.load(f)
        with open(self.model_config_fp, "r") as f:
            self.model_config = json.load(f)

    @staticmethod
    def resetconfig():
        os.makedirs(json_path, exist_ok=True)
        io_config = {"reset": True,
                     "train_prefix": "./train_wav",
                     "test_prefix": "./test_wav",
                     "Train_record_name": "TIMIT_train.tfrecord",
                     "Test_record_name": "TIMIT_test.tfrecord"
                     }

        data_config = {"fs": 8000,
                       "dataexpandrate": 1,
                       "audio_length": 16384,
                       "min_valid_audio_length": 1e4,
                       "max_mix_num": 2,
                       "output_dir": "./output/"
                       }
        train_config = {"lr": 1e-4,
                        "optimizer": "adma",
                        "ckpt_dir": './training_checkpoints/',
                        "epoches": 100,
                        "batch_size": 1,
                        "evaluation_size": 1,
                        "max_to_keep": 5,
                        "plot_pertire": 5000,
                        "savemodel_periter": 50
                        }

        model_config = {"in_channels": 256,
                        "out_channels": 64,
                        "hidden_channels": 128,
                        "K": 200
                        }
        with open(json_path + "/ioconfig.json", "w") as f:
            f.write(json.dumps(io_config))
        with open(json_path + "/data_config.json", "w") as f:
            f.write(json.dumps(data_config))
        with open(json_path + "/train_config.json", "w") as f:
            f.write(json.dumps(train_config))
        with open(json_path + "/model_config.json", "w") as f:
            f.write(json.dumps(model_config))


if __name__ == '__main__':
    ConfigTables.resetconfig()
