# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

# import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob

class Model(object):
    def __init__(self, config):
        print(config)
        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]
        self.subtype_dim = config["subtype_dim"]
        self.num_tags = config["num_tags"]
        self.num_chars = config["num_char"]
        self.num_steps = config["num_steps"]
        self.num_segs = 14
        self.num_subtypes = 51
        self.seq_nums = 8

        self.global_step = tf.Variable(0, trainable = False)
        self.best_dev_f1 = tf.Variable(0.0, trainable = False)
        self.best_test_f1 = tf.Variable(0.0, trainable = False)
        self.initializer = initializers.xavier_initializer()

        self.char_inputs = tf.placeholder(dtype = tf.int32,
                                          shape = [None, None],
                                          name = "ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        self.subtype_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SubInputs")
        self.targets = tf.placeholder(dtype = tf.int32,
                                      shape = [None, None],
                                      name = "Targets")

        self.doc_inputs = tf.placeholder(dtype = tf.int32,
                                      shape = [None, None, self.num_steps],
                                      name = "doc_inputs")
        self.dropout = tf.placeholder(dtype = tf.float32, name = "Dropout")
        self.char_lookup = tf.get_variable(
            name="char_embedding",
            shape=[self.num_chars, self.char_dim],
            initializer=self.initializer)

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices = 1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]

        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, self.subtype_inputs, config)

        doc_embedding = self.doc_embedding_layer(self.doc_inputs, self.lstm_dim, self.lengths, config)

        lstm_inputs = tf.nn.dropout(embedding,self.dropout)

        lstm_outputs, lstm_states = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
        lstm_outputs = tf.nn.dropout(lstm_outputs, self.dropout)

        sen_att_outputs = self.attention(lstm_outputs)

        doc_att_outputs = self.doc_attention(doc_embedding, lstm_states)

        gat_output = self.gate(sen_att_outputs, doc_att_outputs)

        outputs = tf.concat([embedding, gat_output], -1)
        lstm_outputs = self.LSTM_decoder(outputs, self.lstm_dim)

        # lstm_outputs = self.tag_attention(lstm_outputs)
        self.logits = self.project_layer(lstm_outputs)

        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs,seg_inputs, subtype_inputs, config, name = None):
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            if config["subtype_dim"]:
                with tf.variable_scope("subtype_embedding"), tf.device('/cpu:0'):
                    self.subtype_lookup = tf.get_variable(
                        name="subtype_embedding",
                        shape=[self.num_subtypes, self.subtype_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.subtype_lookup, subtype_inputs))
            embed = tf.concat(embedding, axis = -1)
        return embed

    def doc_embedding_layer(self, doc_inputs, lstm_dim, lengths ,config, name = None):

        def doc_LSTM_layer(inputs, lstm_dim, lengths):
            with tf.variable_scope("doc_BiLSTM", reuse=tf.AUTO_REUSE):
                lstm_cell = {}
                for direction in ["doc_forward", "doc_backward"]:
                    with tf.variable_scope(direction):
                        lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                            lstm_dim,
                            use_peepholes=True,
                            initializer=self.initializer,
                            reuse=tf.AUTO_REUSE,
                            state_is_tuple=True
                        )
                (outputs,
                 (encoder_fw_final_state,
                  encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell["doc_forward"],
                    lstm_cell["doc_backward"],
                    inputs,
                    dtype=tf.float32,
                    sequence_length=lengths
                )
                final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), -1)
                return final_state
        lstm_states = []
        doc_inputs = tf.reshape(doc_inputs, [self.batch_size, self.seq_nums, self.num_steps])
        doc_input = tf.unstack(tf.transpose(doc_inputs, [1, 0, 2]), axis=0)
        for i in range(self.seq_nums):
            with tf.variable_scope("doc_embedding", reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
                self.char_doc_lookup = tf.get_variable(
                    name = "doc_embedding",
                    shape = [self.num_chars, self.char_dim],
                    initializer = self.initializer,
                )
                doc_embedding = (tf.nn.embedding_lookup(self.char_doc_lookup, doc_input[i]))
            lstm_state = doc_LSTM_layer(doc_embedding, lstm_dim, lengths)
            lstm_states.append(lstm_state)
        last_states = tf.transpose(lstm_states, [1, 0, 2])
        last_states = tf.reshape(last_states,[self.batch_size,self.seq_nums,self.lstm_dim * 2])
        return last_states

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name = None):
        with tf.variable_scope("char_BiLSTM", reuse=tf.AUTO_REUSE):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes = True,
                        initializer = self.initializer,
                        state_is_tuple = True
                    )
            (outputs,
             (encoder_fw_final_state,
              encoder_bw_final_state))= tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype = tf.float32,
                sequence_length = lengths
            )
            final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), -1)
            return tf.concat(outputs, axis=2), final_state

    def attention(self, lstm_outputs, name=None):
        def bilinear_attention(source, target):
            dim1 = int(source.get_shape()[1])
            seq_size = int(target.get_shape()[1])
            dim2 = int(target.get_shape()[2])
            with tf.variable_scope("att", reuse=tf.AUTO_REUSE):
                W = tf.get_variable("att_W",
                                    shape=[dim1, dim2],
                                    dtype = tf.float32,
                                    initializer=self.initializer,
                                    )

                source = tf.expand_dims(tf.matmul(source, W), 1)
                prod = tf.matmul(source, target, adjoint_b=True)

                prod = tf.reshape(prod, [-1, seq_size])
                prod = tf.tanh(prod)

                alpha = tf.nn.softmax(prod)
                probs3dim = tf.reshape(alpha, [-1, 1, seq_size])
                Bout = tf.matmul(probs3dim, target)
                Bout2dim = tf.reshape(Bout, [-1, dim2])
                return Bout2dim, alpha

        with tf.variable_scope("attention" if not name else name):
            hidden_dim = self.lstm_dim * 2
            sequence_length = self.num_steps
            lstm_outputs = tf.reshape(lstm_outputs, [self.batch_size, self.num_steps, hidden_dim])
            outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]), axis=0)
            fina_outputs = list()
            for i in range(sequence_length):
                atten_out, P = bilinear_attention(outputs[i], lstm_outputs)
                fina_outputs.append(atten_out)

            attention_outputs = tf.transpose(fina_outputs, [1, 0, 2])

            output = tf.reshape(attention_outputs, [self.batch_size, sequence_length, hidden_dim])
            return output

    def doc_attention(self,doc_embedding, lstm_states,name = None):
        def bilinear_attention(source, target):
            dim1 = int(source.get_shape()[1])
            seq_size = int(target.get_shape()[1])
            dim2 = int(target.get_shape()[2])
            with tf.variable_scope('doc_attention', reuse=tf.AUTO_REUSE):
                W = tf.Variable(tf.truncated_normal([dim1, dim2], 0, 1.0), tf.float32, name="W_doc_att")
                b = tf.Variable(tf.truncated_normal([1], 0, 1.0), tf.float32, name='b_doc_att')

                source = tf.expand_dims(tf.matmul(source, W), 1)
                prod = tf.add(tf.matmul(source, target, adjoint_b=True), b)

                prod = tf.reshape(prod, [-1, seq_size])
                prod = tf.tanh(prod)
                alpha = tf.nn.softmax(prod)

                probs3dim = tf.reshape(alpha, [-1, 1, seq_size])
                Bout = tf.matmul(probs3dim, target)
                Bout2dim = tf.reshape(Bout, [-1, dim2])
                return Bout2dim, alpha

        with tf.variable_scope("doc_attention" if not name else name):
            hidden_dim = self.lstm_dim * 2
            sequence_length = self.num_steps
            lstm_states = tf.reshape(lstm_states, [self.batch_size, hidden_dim])
            atten_out,p = bilinear_attention(lstm_states, doc_embedding)
            output = tf.reshape(atten_out, [self.batch_size, hidden_dim])
            output = tf.expand_dims(output, 1)
            output = tf.tile(output, [1, sequence_length, 1])
            return tf.reshape(output, [self.batch_size, self.num_steps, hidden_dim])

    def gate(self, sen_att_outputs, doc_att_outputs):
        gate_dim = self.lstm_dim * 4
        with tf.variable_scope("gate_layer"):
            W = tf.get_variable("W",
                                shape=[gate_dim,gate_dim/2],
                                dtype=tf.float32,
                                initializer=self.initializer,
                                )
            b = tf.get_variable("b",
                                shape=[gate_dim/2],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer()
                                )
            input = tf.concat([sen_att_outputs, doc_att_outputs], -1)
            input = tf.reshape(input, shape=[-1, gate_dim])
            gate = tf.add(tf.matmul(input, W), b)
            gate =  tf.sigmoid(tf.abs(gate))
            gate_ = tf.ones_like(gate) - tf.sigmoid(tf.abs(gate))

            output = tf.concat([np.multiply(tf.reshape(sen_att_outputs, [-1, int(gate_dim/2)]), gate),np.multiply(tf.reshape(doc_att_outputs, [-1, int(gate_dim/2)]), gate_)], -1)
            output = tf.reshape(output, [self.batch_size, self.num_steps, gate_dim])
            return output

    def LSTM_decoder(self, lstm_outputs, lstm_dim):
        def project(h_state, lstm_dim):
            with tf.variable_scope("project_layer", reuse=tf.AUTO_REUSE):
                W = tf.get_variable("W",
                                    shape=[lstm_dim,self.lstm_dim],
                                    dtype=tf.float32,
                                    initializer=self.initializer,
                                    )
                b = tf.get_variable("b",
                                    shape=[self.lstm_dim],
                                    dtype=tf.float32,
                                    initializer = tf.zeros_initializer()
                                    )
                y_pre = tf.add(tf.matmul(h_state, W), b)
                tag_pre = tf.cast(tf.argmax(tf.nn.softmax(y_pre), axis=-1), tf.float32)
                return y_pre, tag_pre

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim, forget_bias=0.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout)

        outputs = []
        tag_outputs = []
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        c_state, h_state = init_state
        tag_pre = tf.zeros([self.batch_size, self.lstm_dim])

        with tf.variable_scope("LSTMD", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                h_state = tf.concat([h_state, tag_pre], -1)
                (cell_output, (c_state, h_state)) = lstm_cell(lstm_outputs[:, time_step, :], (c_state, h_state))
                tag_pre, tag_result= project(cell_output, lstm_dim)
                outputs.append(tag_pre)
        outputs = tf.reshape(tf.transpose(outputs, [1,0,2]), [self.batch_size, self.num_steps, self.lstm_dim])
        return outputs

    def tag_attention(self, lstm_outputs, name=None):
        def bilinear_attention(source, target):
            dim1 = int(source.get_shape()[1])
            seq_size = int(target.get_shape()[1])
            dim2 = int(target.get_shape()[2])
            with tf.variable_scope("tag_att", reuse=tf.AUTO_REUSE):
                W = tf.get_variable("tag_att_W",
                                    shape=[dim1, dim2],
                                    dtype = tf.float32,
                                    initializer=self.initializer,
                                    )

                source = tf.expand_dims(tf.matmul(source, W), 1)
                prod = tf.matmul(source, target, adjoint_b=True)

                prod = tf.reshape(prod, [-1, seq_size])
                prod = tf.tanh(prod)

                alpha = tf.nn.softmax(prod)
                probs3dim = tf.reshape(alpha, [-1, 1, seq_size])
                Bout = tf.matmul(probs3dim, target)
                Bout2dim = tf.reshape(Bout, [-1, dim2])
                return Bout2dim, alpha

        with tf.variable_scope("tag_attention" if not name else name):
            hidden_dim = self.lstm_dim
            sequence_length = self.num_steps
            lstm_outputs = tf.reshape(lstm_outputs, [self.batch_size, self.num_steps, hidden_dim])
            outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]), axis=0)
            fina_outputs = list()
            for i in range(sequence_length):
                atten_out, P = bilinear_attention(outputs[i], lstm_outputs)
                fina_outputs.append(atten_out)
            attention_outputs = tf.transpose(fina_outputs, [1, 0, 2])
            output = tf.reshape(attention_outputs, [self.batch_size, sequence_length, hidden_dim])
            return output


    def project_layer(self, lstm_outputs, name = None):
        hidden_dim = self.lstm_dim
        with tf.variable_scope("project" if not name else name):

            with tf.variable_scope("logits"):
                W = tf.get_variable("W",
                                    shape=[hidden_dim, self.num_tags],
                                    dtype=tf.float32,
                                    initializer = self.initializer
                                    )
                b = tf.get_variable("b",
                                    shape=[self.num_tags],
                                    dtype=tf.float32,
                                    initializer = tf.zeros_initializer()
                                    )
                output = tf.reshape(lstm_outputs, shape = [-1, hidden_dim])
                pred = tf.nn.xw_plus_b(output, W, b)

            return tf.reshape(pred, [self.batch_size, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths):
        with tf.variable_scope("loss"):

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=project_logits, labels=self.targets)

            mask = tf.sequence_mask(lengths)
            losses = tf.boolean_mask(losses, mask)

            tag_outputs = tf.sign(tf.abs(tf.cast(self.targets, tf.float32)))
            tag_outputs = tf.boolean_mask(tag_outputs, mask)

            ones = tf.ones_like(tag_outputs)
            tag_outputs_ = ones - tag_outputs
            alpah = 5
            losses = alpah * np.multiply(losses, tag_outputs) + np.multiply(losses, tag_outputs_)
            loss = tf.reduce_mean(losses)
            return loss


    def create_feed_dict(self, is_train, batch):
        strs, doc_id, chars, segs, subtypes, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs : np.asarray(segs),
            self.subtype_inputs: np.asarray(subtypes),
            self.dropout: 1.0,
            self.doc_inputs:np.asarray(doc_id)
        }
        # self.doc_inputs: np.asarray(doc_id)
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths):
        paths = []
        for score, length in zip(logits, lengths):
            score = score[:length]
            path = tf.cast(tf.argmax(score, axis= -1), tf.int32).eval()
            paths.append(path[0:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        results = []
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths)

            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
