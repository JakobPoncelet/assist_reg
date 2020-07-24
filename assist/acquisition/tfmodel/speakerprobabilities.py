'''@file speakerprobabilities.py
contains speaker probabilities and trainable weights'''

from __future__ import division
import tensorflow as tf

class SpeakerProbs(tf.layers.Layer):
    '''trainable weights for identifying speaker identity probabilities'''

    def __init__(
            self, nr_spk, capsule_dim,
            spkweights_initializer=None,
            spkbias_initializer=None,
            trainable=True,
            name=None,
            **kwargs):

        super(SpeakerProbs, self).__init__()
            # trainable=trainable,
            # name=name)  # **kwargs


        self.nr_spk = nr_spk
        self.capsule_dim = capsule_dim
        #self.spkweights_initializer = spkweights_initializer or tf.zeros_initializer()
        self.spkweights_initializer = spkweights_initializer or tf.initializers.random_uniform()
	    #self.spkweights_initializer = spkweights_initializer or tf.zeros_initializer()

        self.spkbias_initializer = spkbias_initializer or tf.zeros_initializer()

    def build(self, input_shape):

        outputcapsdim = input_shape[-1].value

        self.spkweights = self.add_variable(
            name='spkweights',
            dtype=self.dtype or tf.float32,
            shape=[self.nr_spk, outputcapsdim],
            initializer=self.spkweights_initializer,
            trainable=True
        ) #  shape = [self.capsule_dim, self.nr_spk],

        self.spkbias = self.add_variable(
            name='spkbias',
            dtype=self.dtype or tf.float32,
            shape=[self.nr_spk, 1],
            initializer=self.spkbias_initializer,
            trainable=True
        )

        super(SpeakerProbs, self).build(input_shape)

    def call(self, inputs):
        ''' function call -- inputs: average_capsules, [batch_size x output_dim] '''

        spklogits = self.apply_weights(inputs)
        #speakerprobs = tf.nn.softmax(outputs,0)

        return spklogits

    def apply_weights(self,avg_caps):
        ''' get speakerprobabilities

        args:
            avg_caps [batch_size x output_dim]   shape is (16,2)
            spkweight [nr_spk x output_dim]      shape is (11,2)
        return:
            weights applied to avg_caps
            output [nr_spk x batch_size]
        '''
        with tf.name_scope('speakerprobs'):

            output = tf.matmul(self.spkweights, tf.expand_dims(avg_caps[0, :], 1)) + self.spkbias # expand_dims om van shape (2,) naar (2,1) te gaan
            for i in range(1, avg_caps.shape[0]):
                result = tf.matmul(self.spkweights, tf.expand_dims(avg_caps[i, :], 1)) + self.spkbias
                output = tf.concat([output, result], 1)
        return tf.transpose(output)
