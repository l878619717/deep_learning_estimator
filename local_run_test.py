# !/usr/bin/env python
# coding=utf-8

import sys
import time

import tensorflow as tf

import utils.data_loader as data_load
import utils.model_op as op
import utils.my_utils as my_utils
from models import dnn, dnn_pool, dnn_cate, deepfm, deepfm_pool, deepfm_cate, din, dinfm, dien, dnn_autoint, dnn_emb

my_utils.venus_set_environ()
parse_dict = my_utils.arg_parse(sys.argv)


class ModelParams:
    def __init__(self):
        self.learning_rate_decay_steps = int(parse_dict.get("learning_rate_decay_steps", "10000000"))
        self.learning_rate_decay_rate = float(parse_dict.get("learning_rate_decay_steps", "0.9"))
        # self.hidden_units = [int(i) for i in parse_dict.get("hidden_units", "").split(",")]
        self.max_data_size = int(parse_dict.get("max_data_size", "8000000"))
        self.learning_rate = float(parse_dict.get("learning_rate", "0.001"))
        self.l2_reg = float(parse_dict.get("l2_reg", "0.00001"))
        self.batch_size = int(parse_dict.get("batch_size", "1024"))
        self.embedding_size = int(parse_dict.get("embedding_size", "64"))
        self.cate_feats_size = int(parse_dict.get("cate_feats_size", "200000"))
        self.num_batch_size = int(parse_dict.get("num_batch_size", "100"))

        self.autoint_layer_num = int(parse_dict.get("autoint_layer_num", "2"))
        self.autoint_emb_size = int(parse_dict.get("autoint_emb_size", "32"))
        self.autoint_head_num = int(parse_dict.get("autoint_head_num", "2"))
        self.autoint_use_res = int(parse_dict.get("autoint_use_res", "1"))

        self.action_type = parse_dict.get("action_type", "train")
        self.epochs = int(parse_dict.get("epochs", "3"))
        self.alg_name = parse_dict.get("alg_name", "dinfm")
        self.feat_conf_path = parse_dict.get("feat_conf_path", "files/conf/deepfm")
        self.train_path = parse_dict.get("train_path", "files/data/deepfm/train/")
        self.predict_path = parse_dict.get("predict_path", "files/data/deepfm/pred/")
        self.model_pb = parse_dict.get("model_pb", "files/model_save_pb/" + self.alg_name)
        self.model_dir = parse_dict.get("model_dir", "files/model_save_dir/" + self.alg_name)

        self.hidden_units = [1024, 512, 256]
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.dropout_keep_fm = [1, 1, 1, 1, 1]
        self.is_GPU = 1
        self.num_cpu = 10
        self.log_step_count_steps = 500
        self.save_checkpoints_steps = 500
        self.save_summary_steps = 500
        self.keep_checkpoint_max = 2
        self.rate = 1.0

        self.cont_field_size, self.vector_feats_size, self.cate_field_size, self.multi_feats_size, \
            self.multi_feats_range, self.attention_feats_size, self.attention_range = my_utils.feat_size(self.feat_conf_path, self.alg_name)


def main(_):
    params = ModelParams()

    for key, value in params.__dict__.items():
        print(key, "=", value)

    if params.alg_name == "dnn":
        model = dnn.model_estimator(params)
    elif params.alg_name == "dnn_pool":
        model = dnn_pool.model_estimator(params)
    elif params.alg_name == "dnn_cate":
        model = dnn_cate.model_estimator(params)
    elif params.alg_name == "deepfm":
        model = deepfm.model_estimator(params)
    elif params.alg_name == "deepfm_pool":
        model = deepfm_pool.model_estimator(params)
    elif params.alg_name == "deepfm_cate":
        model = deepfm_cate.model_estimator(params)
    elif params.alg_name == "din":
        model = din.model_estimator(params)
    elif params.alg_name == "dinfm":
        model = dinfm.model_estimator(params)
    elif params.alg_name == "dien":
        model = dien.model_estimator(params)
    elif params.alg_name == "dnn_emb":
        model = dnn_emb.model_estimator(params)
    elif params.alg_name == "dnn_autoint":
        model = dnn_autoint.model_estimator(params)
    else:
        model = dnn.model_estimator(params)
        print("alg_name = %s is error" % params.alg_name)
        exit(-1)

    if params.action_type == "train":
        start_time = time.time()

        train_files = data_load.get_file_list(params.train_path)
        predict_files = data_load.get_file_list(params.predict_path)
        print("--------------train------------")
        trained_model_path = op.model_fit(model, params, train_files, predict_files)
        end_time = time.time()
        print("model_save training time: %.2f s" % (end_time - start_time))

        # save model_pb path to a file
        f = tf.gfile.GFile(params.model_pb + "/test", 'w')
        f.write(str(trained_model_path, encoding="utf-8"))

        print("--------------predict------------")
        op.model_predict(trained_model_path, predict_files, params)

    elif params.action_type == "pred":
        print("--------------predict------------")
        predict_files = data_load.get_file_list(params.predict_path)
        op.model_predict('/Users/R.Stalker/PycharmProjects/deep_learing_estimator/files/model_save_pb/deepfm', predict_files, params)

    else:
        print("action_type = %s is error !!!" % params.action_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
