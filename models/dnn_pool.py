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

    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=params.cate_feats_size, embedding_size=params.embedding_size)
    # single_category -> Embedding
    cate_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
    cate_emb = tf.reshape(cate_emb, shape=[-1, params.cate_field_size * params.embedding_size])
    # multi_category -> Embedding
    multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)

    # deep input dense
    dense = tf.concat([cont_feats, vector_feats, cate_emb, multi_cate_emb], axis=1, name='dense_vector')

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

