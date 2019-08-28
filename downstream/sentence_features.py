# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     token_features
   Author :        Xiaosong Zhou
   date：          2019/8/20
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 获取sentence features，即每一个句子的向量，用cls作为句向量
# 因为取cls作为句向量，所以需要用在具体数据集上微调过的模型，否则CLS向量并不好用
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)
import tensorflow as tf
import tokenization
import modeling
import numpy as np
import h5py
import re



# 配置文件
# data_root是模型文件，可以用预训练的，也可以用在分类任务上微调过的模型
data_root = '../chinese_wwm_ext_L-12_H-768_A-12/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
# init_checkpoint = data_root + 'bert_model.ckpt'
# 这样的话，就是使用在具体任务上微调过的模型来做词向量
init_checkpoint = '../model/legal_fine_tune/model.ckpt-4153'
bert_vocab_file = data_root + 'vocab.txt'

# 经过处理的输入文件路径
file_input_x_c_train = '../data/legal_domain/train_x_c.txt'
file_input_x_c_val = '../data/legal_domain/val_x_c.txt'
file_input_x_c_test = '../data/legal_domain/test_x_c.txt'

# embedding存放路径
# emb_file_dir = '../data/legal_domain/emb_fine_tune.h5'

# graph
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

# 每个sample固定为80个句子
SEQ_LEN = 80
# 每个句子固定为128个token
SENTENCE_LEN = 126


def get_batch_data(x):
    """生成批次数据，一个batch一个batch地产生句子向量"""
    data_len = len(x)

    word_mask = [[1] * (SENTENCE_LEN + 2) for i in range(data_len)]
    word_segment_ids = [[0] * (SENTENCE_LEN + 2) for i in range(data_len)]
    return x, word_mask, word_segment_ids


def read_input(file_dir):
    # 从文件中读到所有需要转化的句子
    # 这里需要做统一长度为510
    # input_list = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        input_list = f.readlines()

    # input_list是输入list，每一个元素是一个str，代表输入文本
    # 现在需要转化成id_list
    word_id_list = []
    for query in input_list:
        tmp_word_id_list = []
        quert_str = ''.join(query.strip().split())
        sentences = re.split('。', quert_str)
        for sentence in sentences:
            split_tokens = token.tokenize(sentence)
            if len(split_tokens) > SENTENCE_LEN:
                split_tokens = split_tokens[:SENTENCE_LEN]
            else:
                while len(split_tokens) < SENTENCE_LEN:
                    split_tokens.append('[PAD]')
            # ****************************************************
            # 如果是需要用到句向量，需要用这个方法
            # 加个CLS头，加个SEP尾
            tokens = []
            tokens.append("[CLS]")
            for i_token in split_tokens:
                tokens.append(i_token)
            tokens.append("[SEP]")
            # ****************************************************
            word_ids = token.convert_tokens_to_ids(tokens)
            tmp_word_id_list.append(word_ids)
        word_id_list.append(tmp_word_id_list)
    return word_id_list


# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False
)

# 加载BERT模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
# 获取最后一层和倒数第二层
encoder_last_layer = model.get_sequence_output()
encoder_last2_layer = model.all_encoder_layers[-2]

# 读取数据
token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)

input_train_data = read_input(file_dir=file_input_x_c_train)
input_val_data = read_input(file_dir=file_input_x_c_val)
input_test_data = read_input(file_dir=file_input_x_c_test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_file = h5py.File('../downstream/emb_sentences.h5', 'w')

    # 训练集
    emb_train = []
    for sample in input_train_data:
        # 一个样本（假设有n个句子）就为一个batch
        word_id, mask, segment = get_batch_data(sample)
        feed_data = {input_ids: np.asarray(word_id), input_mask: np.asarray(mask), segment_ids: np.asarray(segment)}
        last2 = sess.run(encoder_last2_layer, feed_dict=feed_data)
        print('******************************************************************')
        print(last2.shape)
        # last2 shape：(seq_len, 50, 768)
        tmp_list = []
        for i in last2:
            tmp_list.append(i[0])
        if len(tmp_list) > SEQ_LEN:
            tmp_list = tmp_list[:SEQ_LEN]
        else:
            while len(tmp_list) < SEQ_LEN:
                pad_vector = [0 for i in range(768)]
                tmp_list.append(pad_vector)

        emb_train.append(tmp_list)
    # 保存
    emb_train_array = np.asarray(emb_train)
    save_file.create_dataset('train', data=emb_train_array)

    # 验证集
    emb_val = []
    for sample in input_val_data:
        # 一个样本（假设有n个句子）就为一个batch
        word_id, mask, segment = get_batch_data(sample)
        feed_data = {input_ids: np.asarray(word_id), input_mask: np.asarray(mask), segment_ids: np.asarray(segment)}
        last2 = sess.run(encoder_last2_layer, feed_dict=feed_data)
        # last2 shape：(seq_len, 50, 768)
        tmp_list = []
        for i in last2:
            tmp_list.append(i[0])
        if len(tmp_list) > SEQ_LEN:
            tmp_list = tmp_list[:SEQ_LEN]
        else:
            while len(tmp_list) < SEQ_LEN:
                pad_vector = [0 for i in range(768)]
                tmp_list.append(pad_vector)

        emb_val.append(tmp_list)
    # 保存
    emb_val_array = np.asarray(emb_val)
    save_file.create_dataset('val', data=emb_val_array)

    # 测试集
    emb_test = []
    for sample in input_test_data:
        # 一个样本（假设有n个句子）就为一个batch
        word_id, mask, segment = get_batch_data(sample)
        feed_data = {input_ids: np.asarray(word_id), input_mask: np.asarray(mask), segment_ids: np.asarray(segment)}
        last2 = sess.run(encoder_last2_layer, feed_dict=feed_data)
        # last2 shape：(seq_len, 50, 768)
        tmp_list = []
        for i in last2:
            tmp_list.append(i[0])
        if len(tmp_list) > SEQ_LEN:
            tmp_list = tmp_list[:SEQ_LEN]
        else:
            while len(tmp_list) < SEQ_LEN:
                pad_vector = [0 for i in range(768)]
                tmp_list.append(pad_vector)

        emb_test.append(tmp_list)
    # 保存
    emb_test_array = np.asarray(emb_test)
    save_file.create_dataset('test', data=emb_test_array)

    save_file.close()

    print(emb_train_array.shape)
    print(emb_val_array.shape)
    print(emb_test_array.shape)

    # 这边目标是接下游CNN任务，因此先写入所有token的embedding，768维
    # 写入shape直接是(N, max_seq_len + 2, 768)
    # 下游需要选用的时候，如果卷积，则去掉头尾使用，如果全连接，则直接使用头部
    # 这里直接设定max_seq_len=510，加上[cls]和[sep]，得到512
    # 写入(n, 512, 768) ndarray到文件，需要用的时候再读出来，就直接舍弃embedding层
