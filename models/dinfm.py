#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf

import utils.model_layer as my_layer
import utils.model_op as op


def get_attention_size(params):
    attention_field_size = 0
    for i in params.attention_range:
        if i[0] == 0:
            attention_field_size += 1
        elif i[0] == 1:
            attention_field_size += i[2][1] - i[2][0]
        else:
            continue
    return attention_field_size


def model_fn(features, labels, mode, params):
    use_deep = True
    use_fm = True
    tf.set_random_seed(2019)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]
    vector_feats = features["vector_feats"]

    single_cate_feats = cate_feats[:, 0:params.cate_field_size]
    multi_cate_feats = cate_feats[:, params.cate_field_size:params.cate_field_size + params.multi_feats_size]
    attention_cate_feats = cate_feats[:, params.cate_field_size + params.multi_feats_size:]

    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_size)]], trainable=False, dtype=tf.int64,
                                   name="cont_feats_index")
    index_max_size = params.cont_field_size + params.cate_feats_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=index_max_size, embedding_size=params.embedding_size)
    fm_first_order_emb = my_layer.emb_init(name='fm_first_order_emb', feat_num=index_max_size, embedding_size=1)

    with tf.name_scope('fm_part'):
        input_field_size = params.cont_field_size + params.cate_field_size + len(params.multi_feats_range)
        cont_index_add = tf.add(cont_feats_index, params.cate_feats_size)

        # FM_first_order [?, input_field_size]
        # cont
        first_cont_emb = tf.nn.embedding_lookup(fm_first_order_emb, ids=cont_index_add)
        first_cont_emb = tf.reshape(first_cont_emb, shape=[-1, params.cont_field_size])
        first_cont_mul = tf.multiply(first_cont_emb, cont_feats)
        # single_cate
        first_single_cate_emb = tf.nn.embedding_lookup(fm_first_order_emb, ids=single_cate_feats)
        first_single_cate_emb = tf.reshape(first_single_cate_emb, shape=[-1, params.cate_field_size])
        # multi_cate
        first_multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, fm_first_order_emb, multi_cate_feats)
        # concat cont & single_cate & multi_cate
        if len(params.multi_feats_range) > 0:
            first_order_emb = tf.concat([first_cont_mul, first_single_cate_emb, first_multi_cate_emb], axis=1)
        else:
            first_order_emb = tf.concat([first_cont_mul, first_single_cate_emb], axis=1)
        first_order = tf.nn.dropout(first_order_emb, params.dropout_keep_fm[0])

        # FM_second_order [?, embedding_size]
        # cont
        second_cont_emb = tf.nn.embedding_lookup(feats_emb, ids=cont_index_add)
        second_cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_size, 1])
        second_cont_mul = tf.multiply(second_cont_emb, second_cont_value)
        # single_cate -> embedding
        second_single_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
        # multi_category -> embedding
        second_multi = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)
        if len(params.multi_feats_range) > 0:
            second_multi_emb = tf.reshape(second_multi, shape=[-1, len(params.multi_feats_range), params.embedding_size])
        # attention -> embedding
        second_attention = attention_alg(params, feats_emb, multi_cate_feats, single_cate_feats, attention_cate_feats)
        second_attention_emb = tf.reshape(second_attention, shape=[-1, get_attention_size(params), params.embedding_size])
        # concat cont & single_cate & multi_cate & attention
        if len(params.multi_feats_range) > 0:
            second_order_emb = tf.concat([second_cont_mul, second_single_emb, second_multi_emb, second_attention_emb], axis=1)
        else:
            second_order_emb = tf.concat([second_cont_mul, second_single_emb, second_attention_emb], axis=1)

        sum_emb = tf.reduce_sum(second_order_emb, 1)
        sum_square_emb = tf.square(sum_emb)
        square_emb = tf.square(second_order_emb)
        square_sum_emb = tf.reduce_sum(square_emb, 1)
        second_order = 0.5 * tf.subtract(sum_square_emb, square_sum_emb)
        second_order = tf.nn.dropout(second_order, params.dropout_keep_fm[1])

        # FM_res [?, self.input_field_size + embedding_size]
        fm_res = tf.concat([first_order, second_order], axis=1)

    with tf.name_scope('deep_part'):
        # cont
        cont_emb = tf.reshape(second_cont_mul, shape=[-1, params.cont_field_size * params.embedding_size])
        # single_cate
        single_cate_emb = tf.reshape(second_single_emb, shape=[-1, params.cate_field_size * params.embedding_size])
        # multi_category -> embedding
        multi_cate_emb = second_multi
        # multi_cate attention
        multi_attention_emb = second_attention
        if len(params.multi_feats_range) > 0:
            deep_res = tf.concat([cont_emb, single_cate_emb, multi_cate_emb, multi_attention_emb, vector_feats], axis=1,
                                 name='dense_vector')
        else:
            deep_res = tf.concat([cont_emb, single_cate_emb, multi_attention_emb, vector_feats], axis=1,
                                 name='dense_vector')
        # deep
        len_layers = len(params.hidden_units)
        for i in range(0, len_layers):
            deep_res = tf.layers.dense(inputs=deep_res, units=params.hidden_units[i], activation=tf.nn.relu)

    with tf.name_scope('deep_fm'):
        if use_fm and use_deep:
            feats_input = tf.concat([fm_res, deep_res], axis=1)
            feats_input_size = input_field_size + params.embedding_size + params.hidden_units[-1]
        elif use_fm:
            feats_input = fm_res
            feats_input_size = input_field_size + params.embedding_size
        elif use_deep:
            feats_input = deep_res
            feats_input_size = params.hidden_units[-1]

        glorot = np.sqrt(2.0 / (feats_input_size + 1))
        deep_fm_weight = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(feats_input_size, 1)), dtype=np.float32)
        deep_fm_bias = tf.Variable(tf.random_normal([1]))

        out = tf.add(tf.matmul(feats_input, deep_fm_weight), deep_fm_bias)

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
