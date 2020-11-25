#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

import utils.model_layer as my_layer
import utils.model_op as op


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2019)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]
    vector_feats = features["vector_feats"]

    single_cate_feats = cate_feats[:, 0:params.cate_field_size]
    multi_cate_feats = cate_feats[:, params.cate_field_size:]
    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_size)]], trainable=False, dtype=tf.int64,
                                   name="cont_feats_index")

    cont_index_add = tf.add(cont_feats_index, params.cate_feats_size)

    index_max_size = params.cont_field_size + params.cate_feats_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=index_max_size, embedding_size=params.embedding_size)

    # cont_feats -> Embedding
    with tf.name_scope("cont_feat_emb"):
        ori_cont_emb = tf.nn.embedding_lookup(feats_emb, ids=cont_index_add, name="ori_cont_emb")
        cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_size, 1], name="cont_value")
        cont_emb = tf.multiply(ori_cont_emb, cont_value)
        autoint_cont = cont_emb
        cont_emb = tf.reshape(cont_emb, shape=[-1, params.cont_field_size * params.embedding_size], name="cont_emb")

    # single_category -> Embedding
    with tf.name_scope("single_cate_emb"):
        cate_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
        autoint_cate = cate_emb
        cate_emb = tf.reshape(cate_emb, shape=[-1, params.cate_field_size * params.embedding_size])

    # multi_category -> Embedding
    with tf.name_scope("multi_cate_emb"):
        multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)
        autoint_multi_cate = tf.reshape(multi_cate_emb, shape=[-1, len(params.multi_feats_range), params.embedding_size])

    # AutoInteracting
    with tf.name_scope("AutoInt"):
        autoint_input = tf.concat([autoint_cont, autoint_cate, autoint_multi_cate], axis=1)
        for i in range(params.autoint_layer_num):
            autoint_input = my_layer.InteractingLayer(num_layer=i, att_emb_size=params.autoint_emb_size, seed=2020,
                                                      head_num=params.autoint_head_num,
                                                      use_res=params.autoint_use_res)(autoint_input)
        autoint_output = tf.layers.Flatten()(autoint_input)

    # deep input dense
    dense = tf.concat([cont_emb, vector_feats, cate_emb, multi_cate_emb], axis=1, name='dense_vector')

    # deep
    len_layers = len(params.hidden_units)
    for i in range(0, len_layers):
        dense = tf.layers.dense(inputs=dense, units=params.hidden_units[i], activation=tf.nn.relu)

    final_input = tf.concat([autoint_output, dense], axis=1)
    out = tf.layers.dense(inputs=final_input, units=1)
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
