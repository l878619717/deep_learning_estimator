#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
# from tensorflow.python.ops.rnn_cell import LSTMCell
# from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import *

import utils.model_layer as my_layer
import utils.model_op as op
from models.rnn import dynamic_rnn


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2019)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]
    vector_feats = features["vector_feats"]

    single_cate_feats = cate_feats[:, 0:params.cate_field_size]
    multi_cate_feats = cate_feats[:, params.cate_field_size:params.cate_field_size + params.multi_feats_size]
    attention_cate_feats = cate_feats[:, params.cate_field_size + params.multi_feats_size:]

    # init_embedding
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=params.cate_feats_size, embedding_size=params.embedding_size)
    # single_category -> embedding
    single_cate_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
    single_cate_emb = tf.reshape(single_cate_emb, shape=[-1, params.cate_field_size * params.embedding_size])
    # attention
    attention_emb = attention_alg(params, feats_emb, multi_cate_feats, single_cate_feats, attention_cate_feats)
    # dien
    dien_emb = dien_alg(params, feats_emb, single_cate_feats, attention_cate_feats)
    # multi_category -> embedding
    multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)
    # deep input dense
    if len(params.multi_feats_range) > 0:
        dense = tf.concat([cont_feats, vector_feats, single_cate_emb, multi_cate_emb, attention_emb, dien_emb], axis=1,
                          name='dense_vector')
    else:
        dense = tf.concat([cont_feats, vector_feats, single_cate_emb, attention_emb, dien_emb], axis=1, name='dense_vector')
    # deep
    len_layers = len(params.hidden_units)
    for i in range(0, len_layers):
        dense = tf.layers.dense(inputs=dense, units=params.hidden_units[i], activation=tf.nn.relu)
    out = tf.layers.dense(inputs=dense, units=1)
    score = tf.identity(tf.nn.sigmoid(out), name='score')
    model_estimator_spec = op.model_optimizer(params, mode, labels, score)
    return model_estimator_spec


def model_estimator(params):
    # shutil.rmtree(conf.model_dir, ignore_errors=True)
    tf.reset_default_graph()
    config = tf.estimator.RunConfig() \
        .replace(session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU}),
                 log_step_count_steps=params.log_step_count_steps,
                 save_checkpoints_steps=params.save_checkpoints_steps,
                 keep_checkpoint_max=params.keep_checkpoint_max,
                 save_summary_steps=params.save_summary_steps)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        model_dir=params.model_dir,
        params=params,
    )
    return model


def dien_alg(params, init_emb, single_feats, attention_feats):
    all_dien_result = None
    name_scope_count = 0
    for info in params.attention_range:
        name_scope_count += 1
        # info = (0, feat_name, (queries_index), (keys_index_start, keys_index_end))
        # info = (1, feat_name, (queries_index_start, queries_index_end), (keys_index_start, keys_index_end))
        match_type = info[0]

        # keys
        attention_feat = attention_feats[:, info[-1][0]:info[-1][1]]
        nonzero_multi_len = tf.count_nonzero(attention_feat, axis=1)
        # keys_length
        attention_emb = tf.nn.embedding_lookup(init_emb, ids=attention_feat)  # [B, T, H]

        if match_type == 0:
            # queries
            cate_emb = tf.nn.embedding_lookup(init_emb, ids=single_feats[:, info[-2]])  # [B, 1, H]
            cate_emb = tf.reshape(cate_emb, shape=[-1, params.embedding_size])  # [B, H]

            name_scope_name = 'dien_' + str(name_scope_count)
            # dien
            dien_result = model_din_v2_gru_vec_attgru(cate_emb, attention_emb, nonzero_multi_len, params.embedding_size * 2,
                                                      name_scope_name)  # [B, D]
            dien_result = tf.reshape(dien_result, shape=[-1, params.embedding_size * 2])  # [B, D]

        else:
            print("attention_range match_type = %d is error!!!" % match_type)
            continue

        if all_dien_result is None:
            all_dien_result = dien_result
        else:
            all_dien_result = tf.concat([all_dien_result, dien_result], axis=1)

    return all_dien_result


def model_din_v2_gru_vec_attgru(query, keys, keys_len, num_units, scope_name):
    # RNN layer(-s)
    with tf.name_scope(scope_name + '_rnn_1'):
        rnn_outputs, _ = dynamic_rnn(GRUCell(num_units), inputs=keys,
                                     sequence_length=keys_len, dtype=tf.float32,
                                     scope=scope_name + "_gru1")
    # Attention layer
    with tf.name_scope(scope_name + 'attention_layer_1'):
        att_outputs, alphas = din_fcn_attention(query, rnn_outputs, keys_len, scope_name,
                                                softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

    with tf.name_scope(scope_name + '_rnn_2'):
        rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(num_units), inputs=rnn_outputs,
                                                 att_scores=tf.expand_dims(alphas, -1),
                                                 sequence_length=keys_len, dtype=tf.float32,
                                                 scope=scope_name + "_gru2")

    # inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
    # inp = tf.concat(
    #     [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
    # self.build_fcn_net(inp, use_dice=True)
    # print(final_state2.get_shape()) [B, D]
    return final_state2


def din_fcn_attention(query, rnn_output, keys_len, scope_name, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, for_cnn=False):
    if isinstance(rnn_output, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        rnn_output = tf.concat(rnn_output, 2)
    if len(rnn_output.get_shape().as_list()) == 2:
        rnn_output = tf.expand_dims(rnn_output, 1)
    if time_major:
        # (T,B,D) => (B,T,D)
        rnn_output = array_ops.transpose(rnn_output, [1, 0, 2])

    # Trainable parameters
    # mask = tf.equal(mask, tf.ones_like(mask))

    # query_size = query.get_shape().as_list()[-1]
    rnn_output_size = rnn_output.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    query = tf.layers.dense(query, rnn_output_size, activation=None, name=scope_name + '_f1' + stag)
    query = prelu(query, scope=scope_name)
    queries = tf.tile(query, [1, tf.shape(rnn_output)[1]])
    queries = tf.reshape(queries, tf.shape(rnn_output))
    din_all = tf.concat([queries, rnn_output, queries - rnn_output, queries * rnn_output], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name=scope_name + 'f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name=scope_name + 'f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name=scope_name + 'f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(rnn_output)[1]])
    scores = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_len, tf.shape(rnn_output)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not for_cnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, rnn_output)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(rnn_output)[1]])
        output = rnn_output * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(rnn_output))
    if return_alphas:
        return output, scores
    return output


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def attention_alg(params, init_emb, multi_feats, single_feats, attention_feats):
    all_attention_result = None
    for info in params.attention_range:
        # info = (0, feat_name, (queries_index), (keys_index_start, keys_index_end))
        # info = (1, feat_name, (queries_index_start, queries_index_end), (keys_index_start, keys_index_end))
        match_type = info[0]

        # keys

        attention_feat = attention_feats[:, info[-1][0]:info[-1][1]]
        nonzero_multi_len = tf.count_nonzero(attention_feat, axis=1)
        # keys_length
        attention_emb = tf.nn.embedding_lookup(init_emb, ids=attention_feat)  # [B, T, H]

        if match_type == 0:
            # queries
            cate_emb = tf.nn.embedding_lookup(init_emb, ids=single_feats[:, info[-2]])  # [B, 1, H]
            cate_emb = tf.reshape(cate_emb, shape=[-1, params.embedding_size])  # [B, H]

            # attention
            attention_result = attention(cate_emb, attention_emb, nonzero_multi_len)  # [B, 1, H]
            attention_result = tf.reshape(attention_result, shape=[-1, params.embedding_size])  # [B, H]

        elif match_type == 1:
            # queries
            cate_emb = tf.nn.embedding_lookup(init_emb, ids=multi_feats[:, info[-2][0]:info[-2][1]])  # [B, N, H]

            # attention
            attention_result = attention_multi_items(cate_emb, attention_emb, nonzero_multi_len)  # [B, N, 1, H]
            attention_result = tf.reshape(attention_result,
                                          shape=[-1, (info[-2][1] - info[-2][0]) * params.embedding_size])  # [B, N*H]

        else:
            print("attention_range match_type = %d is error!!!" % match_type)
            continue

        if all_attention_result is None:
            all_attention_result = attention_result
        else:
            all_attention_result = tf.concat([all_attention_result, attention_result], axis=1)

    return all_attention_result


def attention(queries, keys, keys_length):
    """
      queries:     [B, H] 前面的B代表的是batch_size，H代表向量维度。
      keys:        [B, T, H] T是一个batch中，当前特征最大的长度，每个样本代表一个样本的特征
      keys_length: [B]
    """
    # H 每个query词的隐藏层神经元是多少，也就是H
    queries_hidden_units = queries.get_shape().as_list()[-1]
    # tf.tile为复制函数，1代表在B上保持一致，tf.shape(keys)[1] 代表在H上复制这么多次, 那么queries最终shape为(B, H*T)
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    # queries.shape(B, T, H) 其中每个元素(T,H)代表T行H列，其中每个样本中，每一行的数据都是一样的
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    # 下面4个变量的shape都是(B, T, H)，按照最后一个维度concat，所以shape是(B, T, H*4), 在这块就将特征中的每个item和目标item连接在了一起
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # (B, T, 80)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    # (B, T, 40)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    # (B, T, 1)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    # (B, 1, T)
    # 每一个样本都是 [1,T] 的维度，和原始特征的维度一样，但是这时候每个item已经是特征中的一个item和目标item混在一起的数值了
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask，每一行都有T个数字，keys_length长度为B，假设第1 2个数字是5,6，那么key_masks第1 2行的前5 6个数字为True
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    # 创建一个和outputs的shape保持一致的变量，值全为1，再乘以(-2 ** 32 + 1)，所以每个值都是(-2 ** 32 + 1)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  # T，根据特征数目来做拉伸
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs


def attention_multi_items(queries, keys, keys_length):
    """
      queries:     [B, N, H] N is the number of ads
      keys:        [B, T, H]
      keys_length: [B]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries_nums = queries.get_shape().as_list()[1]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])  # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
    # print outputs.get_shape().as_list()
    # print keys.get_sahpe().as_list()
    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
    return outputs
