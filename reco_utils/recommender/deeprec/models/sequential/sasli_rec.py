# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["SASLI_RECModel"]


class SASLI_RECModel(SLI_RECModel):

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.attn_loss = self._compute_attn_loss()
        self.loss = tf.add(self.attn_loss, tf.add(self.data_loss, self.regular_loss))
        return self.loss

    def _compute_attn_loss(self):
        attn_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.squared_difference(
                        tf.reshape(self.attn_pred, [-1]),
                        tf.reshape(self.iterator.attn_labels, [-1]),
                    )
                )
            )
        attn_loss = tf.multiply(self.hparams.attn_loss_weight, attn_loss)
        return attn_loss

    def _add_summaries(self):
        tf.compat.v1.summary.scalar("data_loss", self.data_loss)
        tf.compat.v1.summary.scalar("regular_loss", self.regular_loss)
        tf.compat.v1.summary.scalar("attn_loss", self.attn_loss)
        tf.compat.v1.summary.scalar("loss", self.loss)
        merged = tf.compat.v1.summary.merge_all()
        return merged

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output, self.attn_pred = self._build_seq_graph()
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return logit

    def _build_seq_graph(self):
        """The main function to create sli_rec model.
        
        Returns:
            obj:the output of sli_rec section.
        """
        hparams = self.hparams
        with tf.variable_scope("sli_rec"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                att_fea1 = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("att_fea1", att_fea1)

            item_history_embedding_new = tf.concat(
                [
                    self.item_history_embedding,
                    tf.expand_dims(self.iterator.time_from_first_action, -1),
                ],
                -1,
            )
            item_history_embedding_new = tf.concat(
                [
                    item_history_embedding_new,
                    tf.expand_dims(self.iterator.time_to_now, -1),
                ],
                -1,
            )
            with tf.variable_scope("rnn"):
                rnn_outputs, final_state = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
                tf.summary.histogram("LSTM_outputs", rnn_outputs)

            with tf.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, rnn_outputs
                )
                att_fea2 = tf.reduce_sum(att_outputs2, 1)
                tf.summary.histogram("att_fea2", att_fea2)

            # ensemble
            with tf.name_scope("alpha"):
                concat_all = tf.concat(
                    [
                        self.target_item_embedding,
                        att_fea1,
                        att_fea2,
                        tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                    ],
                    1,
                )
                last_hidden_nn_layer = concat_all
                alpha_logit = self._fcn_net(
                    last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                )
                alpha_output = tf.sigmoid(alpha_logit)
                user_embed = att_fea1 * alpha_output + att_fea2 * (1.0 - alpha_output)
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output, alpha_output
