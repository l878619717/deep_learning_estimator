#!/usr/bin/env python
# coding=utf-8
import sys
import time
import shutil

import tensorflow as tf

import utils.data_loader as data_load
import utils.model_op as op
import utils.my_utils as my_utils
from models import dnn, deepfm, dnn_pool, din, deepfm_pool


class ModelParams:
    def __init__(self):
        self.dt = sys.argv[1]
        self.alg_name = sys.argv[2]
        self.mode = sys.argv[3]
        self.epochs = int(sys.argv[4])
        self.embedding_size = int(sys.argv[5])
        self.cate_feats_size = int(sys.argv[6])
        self.feat_conf_path = sys.argv[7]
        self.train_path = sys.argv[8] + self.dt + "/"
        self.predict_path = sys.argv[9] + self.dt + "/"
        self.model_pb = sys.argv[10] + self.dt + "/"
        self.model_dir = sys.argv[11]
        self.is_GPU = int(sys.argv[12])
        hidden_units_str = sys.argv[13]
        self.hidden_units = list()  # self.hidden_units = [1024, 512, 256]
        self.learning_rate = 0.001
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.dropout_keep_fm = [1, 1, 1, 1, 1]
        self.learning_rate_decay_steps = 10000000
        self.learning_rate_decay_rate = 0.9
        self.l2_reg = 0.00001
        self.batch_size = 1024
        self.num_cpu = 20
        self.log_step_count_steps = 500
        self.save_checkpoints_steps = 100
        self.save_summary_steps = 500
        self.keep_checkpoint_max = 3

        self.cont_field_size, self.vector_feats_size, self.cate_field_size, self.multi_feats_size, \
            self.multi_feats_range, self.attention_feats_size, self.attention_range = my_utils.feat_size(self.feat_conf_path, self.alg_name)

        hidden_arr = hidden_units_str.split(",")
        for i in range(len(hidden_arr)):
            self.hidden_units.append(int(hidden_arr[i]))


def main(_):
    params = ModelParams()

    print("---delete old data...")
    delete_dt = my_utils.shift_hour_time(params.dt, -24)
    print("---delete_dt:", delete_dt)
    print(params.train_path[:-11] + delete_dt)
    print(params.predict_path[:-11] + delete_dt)
    shutil.rmtree(params.train_path[:-11] + delete_dt, ignore_errors=True)
    shutil.rmtree(params.predict_path[:-11] + delete_dt, ignore_errors=True)
    # shutil.rmtree('/cephfs/group/omg-qqcom-pac/zijingrong/din_hours/model_dir', ignore_errors=True)
    # exit(-1)

    for key, value in params.__dict__.items():
        print(key, "=", value)

    if params.alg_name == "dnn":
        model = dnn.model_estimator(params)
    elif params.alg_name == "deepfm":
        model = deepfm.model_estimator(params)
    elif params.alg_name == "deepfm_pool":
        model = deepfm_pool.model_estimator(params)
    elif params.alg_name == "din":
        model = din.model_estimator(params)
    elif params.alg_name == "dnn_pool":
        model = dnn_pool.model_estimator(params)
    else:
        model = dnn.model_estimator(params)
        print("alg_name = %s is error" % params.alg_name)
        exit(-1)

    if params.mode == "train":
        start_time = time.time()

        train_files = data_load.get_file_list(params.train_path)
        predict_files = data_load.get_file_list(params.predict_path)
        print("--------------train------------")
        trained_model_path = op.model_fit(model, params, train_files, predict_files)
        end_time = time.time()
        print("model_save training time: %.2f s" % (end_time - start_time))

        # save model_pb path to a file
        f = tf.gfile.GFile(params.model_pb[:-11] + "latest_model_path", 'w')
        f.write(str(trained_model_path, encoding="utf-8"))

        print("--------------predict------------")
        op.model_predict(trained_model_path, predict_files, params)

    elif params.mode == "eval":
        print("--------------predict------------")
        predict_files = data_load.get_file_list(params.predict_path)
        op.model_predict(params.model_pb, predict_files, params)
    else:
        print("action_type = %s is error !!!" % params.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
