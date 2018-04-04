# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob,scope = "RNNEncoder"):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        self.scope = scope

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(self.scope):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class CharCNNEncoder(object):
    """
    The purpose of this module is to take the character level embedding and apply a 1D CNN on it to obtain the final embeddings
    """

    def __init__(self,embed_size,num_filters,kernel_size,padding):
        """
        Inputs:
            embed_size: Size of the character embeddings fed to CNN. Can be thought of as depth of input volume
            num_filters: Number of filter or kerneles to use in for the convolutions
            kernel_size: Size of the kernel to use for convolutions
            padding: Type of padding for convolutions
        """

        self.embed_size = embed_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self, inputs):
        """
        Inputs:
            inputs: character level embeddings for a piece of text
        Outputs:

        """
        with vs.variable_scope("CharCNNEncoder",reuse = tf.AUTO_REUSE):
            conv = tf.layers.conv1d(inputs,self.num_filters,self.kernel_size,padding = self.padding, kernel_initializer = self.initializer, activation = tf.nn.relu)
            _,w,d = conv.get_shape().as_list()    
            pool = tf.layers.max_pooling1d(conv,pool_size = w,strides = 1)
            return pool


class BiDirLSTMEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiDirLSTMEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDAF():
    def __init__(self,keep_prob,h):
        self.keep_prob = keep_prob
        self.h = h
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self,ques,ques_mask,context,context_mask):
        bs,M,d = ques.get_shape().as_list()
        _,N,_ = context.get_shape().as_list()
        context_aug = tf.tile(tf.reshape(context,shape=[-1,N,1,d]),[1,1,M,1])
        ques_aug = tf.tile(tf.reshape(ques,shape=[-1,1,M,d]),[1,N,1,1])
        context_flat = tf.transpose(tf.contrib.layers.flatten(tf.transpose(context_aug,[3,1,2,0])),[1,0])
        ques_flat = tf.transpose(tf.contrib.layers.flatten(tf.transpose(ques_aug,[3,1,2,0])),[1,0])
        prod = context_flat*ques_flat
        concat = tf.transpose(tf.concat([context_flat,ques_flat,prod],axis = 1),[1,0])
        Wsim = tf.get_variable('Wsim',shape=[1,6*self.h],initializer=self.initializer)
        bsim = tf.get_variable('bsim',shape=[1],initializer=tf.zeros_initializer())
        sim_matrix = tf.matmul(Wsim,concat) + bsim
        sim_matrix = tf.reshape(sim_matrix,shape=[-1,N,M])
        #context to question attention
        _,attn_dist = masked_softmax(sim_matrix,ques_mask,2)
        attn_outputs = tf.matmul(attn_dist,ques)
        #question to context attention
        m = tf.reduce_max(sim_matrix,axis = 2)
        _,cont_attn_dist = masked_softmax(m,context_mask,1)
        cont_attn_dist = tf.reshape(cont_attn_dist,shape=[-1,1,N])
        cont_attn_output = tf.matmul(cont_attn_dist,context)
        cont_attn_output = tf.tile(cont_attn_output,[1,N,1])
        #concatenating context, c2q attention and q2c attention
        output = tf.concat([context,attn_outputs,cont_attn_output],axis = 2)
        output = tf.nn.dropout(output,self.keep_prob)

        return output

class CoAttn():
    def __init__(self,keep_prob,h):
        self.keep_prob = keep_prob
        self.h = h
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self,ques,ques_mask,context,context_mask):
        _,M,d = ques.get_shape().as_list()
        _,N,_ = context.get_shape().as_list()
        bs = tf.shape(ques)[0]
        qp = tf.contrib.fully_connected(ques,num_outputs=d,activation_fn=tf.nn.tanh)
        cphi = tf.get_variable('cphi',shape = [1,1,d], initializer = self.initializer)
        cphi = tf.tile(cphi,[bs,1,1])
        qphi = tf.get_variable('qphi',shape = [1,1,d], initializer = self.initializer)
        qphi = tf.tile(qphi,[bs,1,1])
        context = tf.concat([context,cphi],axis = 1)
        qp = tf.concat([qp,qphi],axis = 1)
        L = tf.matmul(context,tf.transpose(qp,[0,2,1]))
        maskphi = tf.zeros([bs,1])
        ques_mask = tf.concat([ques_mask,maskphi],axis = 1)
        context_mask = tf.concat([context_mask,maskphi],axis=1)
        _,alpha = masked_softmax(L,ques_mask,2)
        a = tf.matmul(alpha,qp)
        _,beta = masked_softmax(L,context_mask, 1)
        b = tf.matmul(tf.transpose(beta,[0,2,1]),context)
        s = tf.matmul(alpha,b)
        cnct = tf.concat([s,a],axis = 2)
        return cnct

'''
class GatedAttention():
    def __init__(self, keep_prob, query_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.value_vec_size = value_vec_size
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self,ques,ques_mask,context, context_mask):
        W_u_p = tf.get_variable('W_u_p',shape=[self.query_vec_size,self.query_vec_size],initializer = self.initializer)
        W_u_q = tf.get_variable('W_u_q',shape=[self.value_vec_size,self.value_vec_size],initializer = self.initializer)
        W_v_p = tf.get_variable('W_v_p',shape=[self.query_vec_size,self.query_vec_size],initializer = self.initializer)

'''


class SelfAttn():
    def __init__(self,keep_prob,value_vec_size):
        self.keep_prob = keep_prob
        self.value_vec_size = value_vec_size
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self,values,values_mask):
        #Getting the shape of values
        _,N,d = values.get_shape().as_list()

        #Declaring variables to be used for self attention
        W1 = tf.get_variable('W1',[d,d],initializer = self.initializer)
        b1 = tf.get_variable('b1',[d],initializer = tf.zeros_initializer())
        W2 = tf.get_variable('W2',[d,d],initializer = self.initializer)
        b2 = tf.get_variable('b2',[d],initializer = tf.zeros_initializer())

        #Converting values which is a 3 dimensional tensor to a 2d matrix, to support matmul with a weight matrix
        values_flat = tf.reshape(values,[-1,d])

        #Performing matrix multipication
        a1 = tf.matmul(values_flat,W1) + b1
        a2 = tf.matmul(values_flat,W2) + b2

        #Converting a1,a2 back to 3d tensor shape
        a1 = tf.reshape(a1,[-1,N,d])
        a2 = tf.reshape(a2,[-1,N,d])

        #Tiling a1, a2 appropriately so to add all combinations of a1i and a2j
        a1 = tf.tile(tf.reshape(a1,[-1,N,1,d]),[1,1,N,1])
        a2 = tf.tile(tf.reshape(a2,[-1,1,N,d]),[1,N,1,1])

        #Summing and applying tanh nonlinearity on a1,a2
        a = tf.nn.tanh(a1+a2)

        #Declaring vector v which is to be used for self attention
        v = tf.get_variable('v',[d,1],initializer = self.initializer)

        #Flattening out a to perform matrix multipication with v
        a = tf.reshape(a,[-1,d])

        #Multiplying with v
        E = tf.matmul(a,v)
        E = tf.reshape(E,[-1,N,N])
        print(E.shape)

        #Applying softmax on E
        _,alpha = masked_softmax(E,values_mask,2)
        print(alpha.shape)
        #Taking the weighted sum over all the context vectors
        A = tf.matmul(alpha,values)

        return A

class PointerGRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self,num_units,N, mask):
        self._num_units = num_units
        self.N = N
        self.mask = mask

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self.N

    def __call__(self, inputs, state, scope = None):
        _,d = inputs.get_shape().as_list()
        N = self.N
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("preRNN"):
                W_h_p = tf.get_variable("W_h_p",shape = [d,d])
                W_h_a = tf.get_variable("W_h_a",shape = [d,d])
                v = tf.get_variable("v", shape = [d,1])
                a1 = tf.matmul(inputs,W_h_p)
                a2 = tf.matmul(state,W_h_a)
                a2 = tf.reshape(tf.tile(tf.reshape(a2,[-1,1,d]),[1,N,d]),[-1,d])
                a = tf.tanh(a1+a2)
                s = tf.matmul(a,v)
                s = tf.reshape(s,[-1,self.N])
                _,alpha = masked_softmax(s,self.mask,1)
                c = tf.matmul(alpha,s)

            with tf.variable_scope("Gates"):
                ru = tf.contrib.layers.fully_connected(c,2*self._num_units) + tf.contrib.layers.fully_connected(state,2*self._num_units)
                ru = tf.nn.sigmoid(ru)
                r,u = tf.split(ru,2,1)
                o = tf.tanh(tf.contrib.layers.fully_connected(r*state,self._num_units) + tf.contrib.layers.fully_connected(c,self._num_units))

            new_state = u*state + (1-u)*o
            return s,new_state


class PointerNetwork():
    def __init__(self, value_vec_size, query_vec_size, keep_prob):
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.value_vec_size = value_vec_size
        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_graph(self, attn_contexts, context_mask, ques, ques_mask):
        with tf.variable_scope('ptrNetwork'):
            '''
            bs,N,d = attn_contexts.get_shape().as_list()
            _,M,_ = ques.get_shape().as_list()
            #Defining parameters for the network
            W_u_q = tf.get_variable("W_u_q", shape = [value_vec_size,value_vec_size], initializer = self.initializer)
            W_v_q = tf.get_variable("W_v_q", shape = [value_vec_size,value_vec_size], initializer = self.initializer)
            v = tf.get_variable("v", shape = [d,1], initializer = self.initializer)
            Vr = tf.get_variable("Vr", shape = [1,d], initializer = self.initializer)
            
            # For the initial state
            ques_flat = tf.reshape(ques,[-1,self.value_vec_size])
            a1 = tf.matmul(ques_flat, W_u_q)

            tiled_Vr = tf.tile(Vr,[M*bs,1])
            a2 = tf.matmul(tiled_Vr,W_u_q)

            a = tf.tanh(a1+a2)

            s = tf.matmul(a,v)
            s = tf.reshape(s,[-1,M])

            _,alpha = masked_softmax(s,ques_mask)
            alpha = tf.reshape(alpha,[-1,1,M])
            r = tf.matmul(alpha,ques)
            r = tf.squeeze(r,axis = [1]) #This will be the hidden state of the pointer network
            h_prev = r
            '''
            bs,N,d = attn_contexts.get_shape().as_list()
            attn_contexts_flat = tf.reshape(attn_contexts,[-1,1,d])
            inputs = tf.concat([attn_contexts_flat,attn_contexts_flat],axis = 1)
            print(inputs.shape)
            ptr_cell = PointerGRUCell(d,N,context_mask)
            logits,_ = tf.nn.dynamic_rnn(ptr_cell,inputs,dtype = tf.float32)
            return logits





def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
