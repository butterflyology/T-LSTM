
# coding: utf-8

# T-LSTM for regression
#
# Modified from Baytas 2017's unsupervised learning [project](https://github.com/illidanlab/T-LSTM/blob/master/TLSTM.py)

# Preliminaries

# In[3]:


#get_ipython().run_line_magic('autosave', '60')
import tensorflow as tf


# I want to write out the specific parts of the T-LSTM model and their equivalent in code:
#
# - Short-term memory
# $$C^{S}_{t-1} =  \textrm{tanh}(W_{d}C_{t-1} + b_{d})$$
# `C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)`
#
#
# - Discounted short-term memory
# $$\hat{C}^{S}_{t-1} = C^{S}_{t-1} \times g(\Delta{t})$$
# `C_ST_dis = tf.multiply(T, C_ST)` - Where `T` is elapsed time
#
#
# - Long-term memory
# $$C^{T}_{t-1} = C_{t-1} + C^{S}_{t-1}$$
#
#
# - Adjusted previous memory
# $$C^{*}_{t-1} = C^{T}_{t-1} + \hat{C}^{S}_{t-1}$$
#
#
# - Forget gate
# $$f_{t} = \sigma(W_{f}x_{t} + U_{f}h_{t-1} + b_{f})$$
#     `f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)`
#
#
# - Input gate
# $$i_{t} = \sigma(W_{i}x_{t} + U_{i}h_{t-1} + b_{i})$$
# `i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)`
#
#
# - Output gate
# $$o_{t} = \sigma(W_{o}x_{t} + U_{o}h_{t-1} + b_{o})$$
# `o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)`
#
#
# - Candidate memory
# $$\tilde{C} = \textrm{tanh}(W_{c}x_{t} + U_{c}h_{t-1} + b_{c}$$
# `C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)`
#
#
# - Current memory
# $$C_{t} = f_{t} \times C^{*}_{t-1} + i_{t} \times \tilde{C}$$
# `Ct = f * prev_cell + i * C`
#
#
# - Current hidden state
# $$h_{t} = o \times \textrm{tanh}(C_{t})$$
# `current_hidden_state = o * tf.nn.tanh(Ct)`
#
#
# **Where**:
#
# - $x_{t}$ represents the current input
# - $h_{t-1}$ and $h_{t}$ are the previous and current hidden states
# - $C_{t-1}$ and $C_{t}$ are the previous and current cell memories
# - $\{W_{f}, U_{f}, b_{f}\}$ - forget gate network parameters
# - $\{W_{i}, U_{i}, b_{i}\}$ - input gate network parameters
# - $\{W_{o}, U_{o}, b_{o}\}$ - output gate network parameters
# - $\{W_{c}, U_{c}, b_{c}\}$ - candidate memory network parameters
# - $\{W_{d}, b_{d}\}$ - subspace decomposition network parameters
# - $\Delta_{t}$ - elapsed time between $x_{t-1}$ and $x_{t}$
# - $g(\cdot)$ is the decay function
#     - $g(\Delta_{t}) = \frac{1}{\Delta_{t}}$ - small amount of elapsed time
#     - $g(\Delta_{t}) = \frac{1}{\textrm{log}(e + \Delta_{t})}$ - large amount of elapsed time

# - $\{W_{f}, U_{f}, b_{f}\}$ - forget gate network parameters
#     - $W_{f}$ - `self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name = 'Forget_Hidden_weight',reg = None)`
#     - $U_{f}$ - `self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Forget_State_weight',reg = None)`
#     - $b_{f}$ - `self.bf = self.init_bias(self.hidden_dim, name = 'Forget_Hidden_bias')`
#
#
# - $\{W_{i}, U_{i}, b_{i}\}$ - input gate network parameters
#     - $W_{i}$ - `self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name = 'Input_Hidden_weight', reg = None)`
#     - $U_{i}$ - `self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Input_State_weight',reg = None)`
#     - $b_{i}$ - `self.bi = self.init_bias(self.hidden_dim, name = 'Input_Hidden_bias')`
#
#
# - $\{W_{o}, U_{o}, b_{o}\}$ - output gate network parameters
#     - $W_{o}$ - `self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name = 'Output_Hidden_weight',reg = None)`
#     - $U_{o}$ - `self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Output_State_weight',reg = None)`
#     - $b_{i}$ - `self.bog = self.init_bias(self.hidden_dim, name = 'Output_Hidden_bias')`
#
#
# - $\{W_{c}, U_{c}, b_{c}\}$ - candidate memory network parameters
#     - $W_{c}$ - `self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name = 'Cell_Hidden_weight',reg = None)`
#     - $U_{c}$ - `self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Cell_State_weight',reg = None)`
#     - $b_{i}$ - `self.bc = self.init_bias(self.hidden_dim, name = 'Cell_Hidden_bias')`
#
#
# - $\{W_{d}, b_{d}\}$ - subspace decomposition network parameters
#     - $W_{d}$ - `self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Decomposition_Hidden_weight')`
#     - $b_{d}$ - `self.b_decomp = self.no_init_bias(self.hidden_dim, name = 'Decomposition_Hidden_bias_enc')`

# In[4]:


class TLSTM(object):
    def init_weights(self, input_dim, output_dim, name, std = 0.1, reg = None):
        return tf.get_variable(name, shape = [input_dim, output_dim], initializer = tf.random_normal_initializer(0.0, std), regularizer = reg)


    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape = [output_dim], initializer = tf.constant_initializer(1.0))


    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name,shape = [input_dim, output_dim])


    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name,shape = [output_dim])


    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim,train):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # [batch size x seq length x input dim]
        self.input = tf.placeholder('float', shape = [None, None, self.input_dim])
        self.labels = tf.placeholder('float', shape = [None, output_dim])
        self.time = tf.placeholder('float', shape = [None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        if train == 1:

            # forget gate network parameters
            self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name = 'Forget_Hidden_weight',reg = None)
            self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Forget_State_weight',reg = None)
            self.bf = self.init_bias(self.hidden_dim, name = 'Forget_Hidden_bias')

            # input gate network parameters
            self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name = 'Input_Hidden_weight', reg = None)
            self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Input_State_weight',reg = None)
            self.bi = self.init_bias(self.hidden_dim, name = 'Input_Hidden_bias')

            # output gate network parameters
            self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name = 'Output_Hidden_weight',reg = None)
            self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Output_State_weight',reg = None)
            self.bog = self.init_bias(self.hidden_dim, name = 'Output_Hidden_bias')

            # candidate memory network parameters
            self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name = 'Cell_Hidden_weight',reg = None)
            self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Cell_State_weight',reg = None)
            self.bc = self.init_bias(self.hidden_dim, name = 'Cell_Hidden_bias')

            # subspace decomposition network parameters
            self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name = 'Decomposition_Hidden_weight',reg = None)
            self.b_decomp = self.init_bias(self.hidden_dim, name = 'Decomposition_Hidden_bias_enc')

            # not sure
            self.Wo = self.init_weights(self.hidden_dim, fc_dim, name = 'Fc_Layer_weight', reg = None) #tf.contrib.layers.l2_regularizer(scale = 0.001)
            self.bo = self.init_bias(fc_dim, name = 'Fc_Layer_bias')

            # softmax for classification
            self.W_softmax = self.init_weights(fc_dim, output_dim, name = 'Output_Layer_weight', reg = None)#tf.contrib.layers.l2_regularizer(scale=0.001)
            self.b_softmax = self.init_bias(output_dim, name = 'Output_Layer_bias')

        else:

            # forget gate network parameters
            self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, name = 'Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Forget_State_weight')
            self.bf = self.no_init_bias(self.hidden_dim, name = 'Forget_Hidden_bias')

            # input gate network parameters
            self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, name = 'Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Input_State_weight')
            self.bi = self.no_init_bias(self.hidden_dim, name = 'Input_Hidden_bias')

            # output gate network parameters
            self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, name = 'Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Output_State_weight')
            self.bog = self.no_init_bias(self.hidden_dim, name = 'Output_Hidden_bias')

            # candidate memory network parameters
            self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, name = 'Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Cell_State_weight')
            self.bc = self.no_init_bias(self.hidden_dim, name = 'Cell_Hidden_bias')

            # subspace decomposition network parameters
            self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name = 'Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.hidden_dim, name = 'Decomposition_Hidden_bias_enc')

            # not sure
            self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, name = 'Fc_Layer_weight')
            self.bo = self.no_init_bias(fc_dim, name = 'Fc_Layer_bias')

            # softmax for classification
            self.W_softmax = self.no_init_weights(fc_dim, output_dim, name = 'Output_Layer_weight')
            self.b_softmax = self.no_init_bias(output_dim, name = 'Output_Layer_bias')


    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis


        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])


    def get_states(self): # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm = [2, 0, 1])
        scan_input = tf.transpose(scan_input_) #scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time) # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0],tf.shape(scan_time)[1],1])
        concat_input = tf.concat([scan_time, scan_input],2) # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.TLSTM_Unit, concat_input, initializer=ini_state_cell, name = 'states')
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states


    def get_output(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self): # Returns all the outputs
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :] # Compatible with tensorflow 1.2.1
        # output = tf.reverse(all_outputs, [True, False, False])[0, :, :] # Compatible with tensorflow 0.12.1
        return output

    def get_cost_acc(self):
        logits = self.get_outputs()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        y_pred = tf.argmax(logits, 1)
        y = tf.argmax(self.labels, 1)
        return cross_entropy, y_pred, y, logits, self.labels


    def map_elapse_time(self, t):

        c1 = tf.constant(1, dtype = tf.float32)
        c2 = tf.constant(2.7183, dtype = tf.float32)

        # T = tf.multiply(self.wt, t) + self.bt

        T = tf.div(c1, tf.log(t + c2), name = 'Log_elapse_time')

        Ones = tf.ones([1, self.hidden_dim], dtype = tf.float32)

        T = tf.matmul(T, Ones)

        return T
