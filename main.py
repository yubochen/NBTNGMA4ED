# encoding = utf-8
import os
import pickle
import itertools
from collections import OrderedDict
import json
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, create_model
from utils import  save_config, load_config
from utils import test_ner, save_model, clean
from utils import make_path, load_config, save_config

from data_utils import BatchManager, load_word2vec
from evaluation_pro import evaluation



flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,          "clean train floder")
flags.DEFINE_boolean("train",       False,          "Wither train the model")

flags.DEFINE_integer("seg_dim",     20,              "Embendding size for type 0 if not used")
flags.DEFINE_integer("subtype_dim", 20,              "Embendding size for subtype 0 if not used")
flags.DEFINE_integer("num_steps",   40,              "num_steps")

flags.DEFINE_integer("char_dim",    100,             "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,             "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iob",           "tagging schema iobes or iob")

flags.DEFINE_float("clip",          5,                  "Gradient clip")
flags.DEFINE_float("dropout",       0.5,                "Dropout rate")
flags.DEFINE_integer("batch_size",    20,                 "batch size")
flags.DEFINE_float("lr",            0.001,               "Initiaal learning rate")
flags.DEFINE_string("optimizer",    "adam",              "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,               "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,              "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,               "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,                "maximum trainning epochs")
flags.DEFINE_integer("steps_check", 100,                "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",              "Path to save model")
flags.DEFINE_string("summary_path", "summary",          "Path to store summaries")
flags.DEFINE_string("log_file",     "train_log",        "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",         "File for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",       "File for vocab")
flags.DEFINE_string("config_file",  "config_file",      "File for config")
flags.DEFINE_string("script",       "conlleval",        "evaluation scropt")
flags.DEFINE_string("result_path",  "result",           "Path for results")
flags.DEFINE_string("emb_file",     "100.utf8",    "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data_doc","example.train"),   "path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data_doc","example.dev"),     "path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data_doc","example.test"),    "path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip <5.1,         "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1,  "dropout rate should between 0 and 1"
assert FLAGS.lr > 0,            "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


def print_config(config, logger):
    with open("result.json", 'a', encoding="utf-8") as outfile:
        for k, v in config.items():
            logger.info("{}:\t{}".format(k.ljust(15), v))
            json.dump("{}:{}".format(k.ljust(15), v), outfile, ensure_ascii=False)
            outfile.write('\n')

def config_model(char_to_id, tag_to_id):
    config = dict()
    config["num_char"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"]  = FLAGS.seg_dim
    config["subtype_dim"]  = FLAGS.subtype_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["num_steps"] = FLAGS.num_steps

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score :{:>.3f}".format(f1))
        return f1 > best_test_f1

def train():
    train_sentences = load_sentences(FLAGS.train_file)
    dev_sentences = load_sentences(FLAGS.dev_file)
    test_sentences = load_sentences(FLAGS.test_file)

    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]

            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower )

        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file,'wb') as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, 'rb') as f:
            char_to_id,id_to_char, tag_to_id, id_to_tag = pickle.load(f)


    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id
    )
    train_manager = BatchManager(train_data, FLAGS.batch_size, FLAGS.num_steps)


    dev_manager   = BatchManager(dev_data, 100, FLAGS.num_steps)
    test_manager  = BatchManager(test_data, 100, FLAGS.num_steps)

    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config,logger)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config = tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(75):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{},".format(iteration, step%steps_per_epoch, steps_per_epoch))
                    loss = []
            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def test():
    make_path(FLAGS)
    config = load_config(FLAGS.config_file)
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    test_sentences = load_sentences(FLAGS.test_file)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id
    )
    test_manager  = BatchManager(test_data, 100, FLAGS.num_steps)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config = tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        test()

if __name__ == "__main__":
    tf.app.run()

