# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from tensorflow.keras import backend as K

__all__ = ["SURGEModel"]


class SURGEModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.relative_threshold = 0.5 
        self.metric_heads = 1
        self.attention_heads = 1
        self.pool_layers = 1
        self.layer_shared = True
        if 'kwai' in socket.gethostname():
            self.pool_length = 150 # kuaishou
        else:
            self.pool_length = 30 # taobao
        super().__init__(hparams, iterator_creator, seed=None)


    def _build_seq_graph(self):
        """ SURGE Model: 

            1) Interest graph: Graph construction based on metric learning
            2) Interest fusion and extraction : Graph convolution and graph pooling 
            3) Prediction: Flatten pooled graph to reduced sequence
        """
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )
        self.mask = self.iterator.mask
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)

        with tf.name_scope('interest_graph'):
            ## Node similarity metric learning 
            S = []
            for i in range(self.metric_heads):
                # weighted cosine similarity
                self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
                X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
                X_fts = tf.nn.l2_normalize(X_fts,dim=2)
                S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
                # min-max normalization for mask
                S_min = tf.reduce_min(S_one, -1, keepdims=True)
                S_max = tf.reduce_max(S_one, -1, keepdims=True)
                S_one = (S_one - S_min) / (S_max - S_min)
                S += [S_one]
            S = tf.reduce_mean(tf.stack(S, 0), 0)
            # mask invalid nodes
            S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)

            ## Graph sparsification via seted sparseness 
            S_flatten = tf.reshape(S, [tf.shape(S)[0],-1])
            if 'kwai' in socket.gethostname():
                sorted_S_flatten = tf.contrib.framework.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            # relative ranking strategy of the entire graph
            num_edges = tf.cast(tf.count_nonzero(S, [1,2]), tf.float32) # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
            if 'kwai' in socket.gethostname():
                threshold_index = tf.stack([tf.range(tf.shape(X)[0]), tf.cast(to_keep_edge, tf.int32)], 1) # B*2
                threshold_score = tf.gather_nd(sorted_S_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                threshold_score = tf.gather_nd(sorted_S_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)


        with tf.name_scope('interest_fusion_extraction'):
            for l in range(self.pool_layers):
                reuse = False if l==0 else True
                X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l, reuse=reuse)


        with tf.name_scope('prediction'):
            # flatten pooled graph to reduced sequence 
            output_shape = self.mask.get_shape()
            if 'kwai' in socket.gethostname():
                sorted_mask_index = tf.contrib.framework.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.contrib.framework.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            sorted_mask.set_shape(output_shape)
            sorted_mask_index.set_shape(output_shape)
            X = tf.batch_gather(X, sorted_mask_index) # B*L*F  < B*L = B*L*F
            self.mask = sorted_mask
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B

            # cut useless sequence tail per batch 
            self.to_max_length = tf.range(tf.reduce_max(self.reduced_sequence_length)) # l
            X = tf.gather(X, self.to_max_length, axis=1) # B*L*F -> B*l*F
            self.mask = tf.gather(self.mask, self.to_max_length, axis=1) # B*L -> B*l
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B

            # use cluster score as attention weights in AUGRU 
            _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(self.hparams.hidden_size),
                inputs=X,
                att_scores = tf.expand_dims(alphas, -1),
                sequence_length=self.reduced_sequence_length,
                dtype=tf.float32,
                scope="gru"
            )
            model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout*self.target_item_embedding], 1)

        return model_output

  
    def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W 
            reuse (obj): Reusing variable W in query operation 
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.variable_scope("attention_fcn"+str(name), reuse=reuse):
            query_size = query.shape[-1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat"+str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])

            if query.shape.ndims != att_inputs.shape.ndims:
                queries = tf.reshape(
                    tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
                )
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = key_value * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights


    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling 

        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation 

        Returns:
            X (obj): Aggerated cluster embedding 
            A (obj): Pooled adjacency matrix 
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer

        """
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            X_q = tf.matmul(A, tf.matmul(A, X)) # B*L*F

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                ## graph attentive convolution
                E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, 40, use_bias=False)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

        with tf.name_scope('interest_extraction'):
            ## cluster fitness score 
            X_q = tf.matmul(A, tf.matmul(A, Xc)) # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                cluster_score += [f_1 + f_2]
            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(
                tf.where(boolean_mask, cluster_score, mask_paddings),
                axis = -1
            )

            ## graph pooling
            num_nodes = tf.reduce_sum(self.mask, 1) # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool, 
                               tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32), 
                               num_nodes)  # B
            cluster_score = cluster_score * self.float_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1) # B*2
                target_score = tf.gather_nd(sorted_score, target_index) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(self.mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1)

            ## ensure graph connectivity 
            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool),
                          tf.transpose(E, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout 
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.float_mask, -1), 1)

        return Xc, A, graph_readout, cluster_score
