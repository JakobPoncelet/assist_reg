'''@file reconstruction.py
contains words to decode to and trainable weights'''

from __future__ import division
import tensorflow as tf

class WordProbs(tf.layers.Layer):
    '''trainable weights for identifying which words are said specifically''' 

    def __init__(
            self, all_words, capsule_dim,
            wordweights_initializer=None,
            wordbias_initializer=None,
            trainable=True,
            name=None,
            **kwargs):

        super(WordProbs, self).__init__()
            # trainable=trainable,
            # name=name)  # **kwargs


        self.all_words = all_words
        self.capsule_dim = capsule_dim
        self.wordweights_initializer = wordweights_initializer or tf.zeros_initializer()
        self.wordbias_initializer = wordbias_initializer or tf.zeros_initializer()

    def build(self, input_shape):

        outputcapsdim = input_shape[-1].value

        self.wordweights = self.add_variable(
            name='wordweights',
            dtype=self.dtype or tf.float32,
            shape=[len(self.all_words), outputcapsdim],
            initializer=self.wordweights_initializer,
            trainable=True
        )

        self.wordbias = self.add_variable(
            name='wordbias',
            dtype=self.dtype or tf.float32,
            shape=[len(self.all_words), 1],
            initializer=self.wordbias_initializer,
            trainable=True
        )

        super(WordProbs, self).build(input_shape)

    def call(self, inputs):
        ''' function call -- inputs: average_capsules, [batch_size x output_dim]
            returns: wordprobs shape=(16,121)
        '''

        wordlogits = self.apply_weights(inputs)
        #wordprobs = tf.sigmoid(wordlogits)

        return wordlogits

    def apply_weights(self,avg_caps):
        ''' get word probabilities

        args:
            avg_caps [batch_size x output_dim]   shape is (16,8)
            wordweight [nr_words x output_dim]   shape is (121,8)
        return:
            weights applied to avg_caps
            output [nr_words x batch_size]       shape is (121,16)
        '''
        with tf.name_scope('wordprobs'):

            output = tf.matmul(self.wordweights, tf.expand_dims(avg_caps[0, :], 1)) + self.wordbias # expand_dims om van shape (2,) naar (2,1) te gaan
            for i in range(1, avg_caps.shape[0]):
                result = tf.matmul(self.wordweights, tf.expand_dims(avg_caps[i, :], 1)) + self.wordbias
                output = tf.concat([output, result], 1)
        return tf.transpose(output)
