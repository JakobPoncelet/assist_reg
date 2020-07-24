'''@file distributor_capsules.py
Contains the DistributorCapsules class'''

import tfmodel
import layers
import ops
import tensorflow as tf
import numpy as np
import speakerprobabilities
import reconstruction

class RCCN_SPK(tfmodel.TFModel):
    '''an encoder-decoder with dynamic routing acquisition model'''

    def model(self, inputs, seq_length, nr_spk, all_words, targets):
        '''apply the model'''

        with tf.variable_scope('model'):

            #encode the features
            encoded, seq_length = self.encoder(inputs, seq_length)

            #compute the primary capsules
            prim_capsules, contrib = self.primary_capsules(encoded, seq_length)

            #get the output_capsules
            output_capsules, alignment, logits = self.output_capsules(
                prim_capsules, contrib)

            #compute the label probabilities
            if self.conf['slot_filling'] == 'True':
                labelprobs, alignment = self.slot_filling(output_capsules, alignment)
            else:
                labelprobs = ops.safe_norm(output_capsules)

            #get the average capsules
            average_capsules, mask, masked_output = self.average_capsules(output_capsules, targets)

            #get the speaker probabilities
            speaker = speakerprobabilities.SpeakerProbs(
                nr_spk=nr_spk,
                capsule_dim=int(self.conf['output_capsule_dim']))

            spklogits = speaker(average_capsules)

            #get the word probabilities after reconstructing
            wordreconstructor = reconstruction.WordProbs(
                all_words=all_words,
                capsule_dim=int(self.conf['output_capsule_dim']))

            wordlogits = wordreconstructor(average_capsules)

            # otherwise storing errors in testing phase
            if targets == None:
                targets = tf.zeros(output_capsules.shape, dtype=tf.float32)

            tf.add_to_collection('image', tf.expand_dims(alignment, 3, 'ali'))
            tf.add_to_collection('store', tf.identity(alignment, 'alignment'))
            tf.add_to_collection(
                'store', tf.identity(output_capsules, 'output_capsules'))
            tf.add_to_collection(
                'store', tf.identity(logits, 'logits'))
            tf.add_to_collection(
                'store', tf.identity(labelprobs, 'label_probabilities'))
            tf.add_to_collection(
                'store', tf.identity(spklogits, 'speaker_logits'))
            tf.add_to_collection(
                'store', tf.identity(wordlogits, 'word_logits'))
            tf.add_to_collection('store', tf.identity(average_capsules, 'average_capsules'))
            tf.add_to_collection('store', tf.identity(targets, 'targets'))
            tf.add_to_collection('store', tf.identity(mask, 'mask'))
            tf.add_to_collection('store', tf.identity(masked_output, 'masked_output'))

        return labelprobs, spklogits, wordlogits

    def loss(self, targets, speakers, correct_words, labelprobs, spklogits, wordlogits, wordfactors_present, wordfactors_absent):
        '''compute the loss

        args:
            targets: the reference targets
            speakers: the reference speaker identities (one-hot encoded)
            labelprobs: the label probabilities
            spkprobs: the speaker probabilities

        returns: the loss'''

        with tf.name_scope('compute_loss'):
            #compute the loss
            iw = float(self.conf['insertion_weight'])
            up = float(self.conf['upper_prob'])
            lp = float(self.conf['lower_prob'])
            iloss = iw*tf.reduce_mean(
                tf.reduce_sum((1-targets)*tf.maximum(labelprobs-lp, 0)**2, 1))
            dloss = tf.reduce_mean(
                tf.reduce_sum(targets*tf.maximum(up-labelprobs, 0)**2, 1))
            labelloss = dloss + iloss

            speakerweight = float(self.conf['speakerweight'])
            spkloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=speakers, logits=spklogits))
            # logits & labels have shape [batch_size x num_labels]
            # returns [batch_size] --> reduce_mean over batch
            #spkloss = tf.reduce_mean(-tf.reduce_sum(speakers * tf.nn.log_softmax(spklogits)), 1)

            wordweight = float(self.conf['wordweight'])
            #wordloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=correct_words, logits=wordlogits))
            zeros = tf.zeros_like(wordlogits,dtype=wordlogits.dtype)
            cond = (wordlogits >= zeros)
            relu_logits = tf.where(cond,wordlogits,zeros)  #MAX(x,0) with x = logits
            neg_abs_logits = tf.where(cond,-wordlogits,wordlogits)  #-abs(x)
            result1 = tf.add(relu_logits,tf.log1p(tf.exp(neg_abs_logits)))  #=max(x,0) + log(1+exp(-abs(x)))
            result2 = tf.subtract(result1,wordlogits)   #=max(x,0) + log(1+exp(-abs(x))) - x
            ones = tf.ones_like(correct_words,dtype=correct_words.dtype)
            inv_labels = tf.subtract(ones,correct_words)  #(1-z) with z = targets/labels
            #prod1 = inv_labels*result1
            #prod2 = correct_words*result2
            #prod3 = wordfactors*prod2
            #total = prod1 + prod3
            factor_diff = tf.subtract(wordfactors_present,wordfactors_absent)  #(a_k - b_k)
            prod1 = tf.add(factor_diff*correct_words, wordfactors_absent)  #(a_k - b_k)*z + b_k
            first_term = prod1*result2
            prod2 = wordlogits*wordfactors_absent  #x*b_k 
            second_term = prod2*inv_labels   #x*b_k*(1-z)
            total = first_term + second_term
            wordloss = tf.reduce_mean(total)
            
            loss = labelloss + speakerweight*spkloss + wordweight*wordloss
            return loss, labelloss, spkloss, wordloss

    def encoder(self, features, seq_length):
        '''encode the input features

        args:
            features: a [N x T x F] tensor
            seq_length: an [N] tensor containing the sequence lengths

        returns:
            - the encoded features
            - the encode features sequence lengths
        '''

        with tf.variable_scope('encoder'):

            encoded = tf.identity(features, 'features')
            seq_length = tf.identity(seq_length, 'input_seq_length')

            for l in range(int(self.conf['numlayers_encoder'])):
                with tf.variable_scope('layer%d' % l):
                    num_units = int(self.conf['numunits_encoder'])
                    fw = tf.contrib.rnn.GRUCell(num_units)
                    bw = tf.contrib.rnn.GRUCell(num_units)
                    encoded, _ = tf.nn.bidirectional_dynamic_rnn(
                        fw, bw, encoded, dtype=tf.float32,
                        sequence_length=seq_length)

                    encoded = tf.concat(encoded, 2)

                    if l != int(self.conf['numlayers_encoder']) - 1:
                        with tf.name_scope('sub-sample'):
                            encoded = encoded[:, ::int(self.conf['subsample'])]
                        seq_length = tf.to_int32(tf.ceil(
                            tf.to_float(seq_length)/
                            float(self.conf['subsample'])))

            encoded = tf.identity(encoded, 'encoded')
            seq_length = tf.identity(seq_length, 'output_seq_length')

        return encoded, seq_length

    def primary_capsules(self, encoded, seq_length):
        '''compute the primary capsules

        args:
            encoded: encoded sequences [batch_size x time x dim]
            seq_length: the sequence lengths [batch_size]

        returns:
            the primary capsules
                [batch_size x num_capsules x capsule_dim]
            the contribution of the time steps in the primary capsules
                [batch_size x time x num_capsules x capsule_dim]
        '''

        with tf.variable_scope('primary_capsules'):

            encoded = tf.identity(encoded, 'encoded')
            seq_length = tf.identity(seq_length, 'seq_length')

            num_capsules = int(self.conf['num_capsules'])
            capsule_dim = int(self.conf['capsule_dim'])

            #distribute timesteps over primary capsules
            distribution = tf.layers.dense(encoded, num_capsules, tf.nn.softmax)

            #put a weight on all the timesteps
            attention = tf.layers.dense(encoded, 1, tf.nn.sigmoid)
            weights = attention*distribution

            #compute the weighted averages
            combinations = tf.matmul(weights, encoded, transpose_a=True)

            #map the averages to primary capsule orientations
            layer = tf.layers.Dense(
                capsule_dim, use_bias=False,
                name='orientations')
            orientations = layer(combinations)

            primary_capsules = ops.squash(orientations)

            #get the squash factor
            squash = tf.get_default_graph().get_tensor_by_name(
                'model/primary_capsules/squash/div_1:0')
            contrib = layer(encoded)
            contrib = (
                tf.expand_dims(squash, 1)
                *tf.expand_dims(contrib, 2)
                *tf.expand_dims(weights, 3))

            tf.add_to_collection(
                'image', tf.expand_dims(weights, 3, 'prim_weights'))

            primary_capsules = tf.identity(primary_capsules, 'primary_capsules')

        return primary_capsules, contrib

    def output_capsules(self, rc_capsules, contrib):
        '''compute the output capsules

        args:
            rc_capsules: the rate coded capsules
                [batch_size x num_capsules x capsule_dim]
            contrib: the conttibution of each timestep in the rc capsules
                [batch_size x time x num_capsules x capsule_dim]

        returns:
            the output_capsules [batch_size x num_capsules x capsule_dim]
            the alignment of the output capsules to the timesteps
                [batch_size x time x num_capsules]
        '''

        with tf.variable_scope('output_capsules'):

            capsules = tf.identity(rc_capsules, 'primary_capsules')
            contrib = tf.identity(contrib, 'contrib')
            r = int(self.conf['capsule_ratio'])**int(self.conf['num_rc_layers'])
            num_capsules = self.num_output_capsules*r
            capsule_dim = int(self.conf['output_capsule_dim'])/r

            for l in range(int(self.conf['num_rc_layers'])):
                with tf.variable_scope('layer%d' % l):

                    num_capsules /= int(self.conf['capsule_ratio'])
                    capsule_dim *= int(self.conf['capsule_ratio'])

                    layer = layers.Capsule(
                        num_capsules=num_capsules,
                        capsule_dim=capsule_dim,
                        routing_iters=int(self.conf['routing_iters'])
                    )

                    capsules = layer(capsules)

                    #get the predictions for the contributions
                    contrib_predict, _ = layer.predict(contrib)

                    #get the final routing logits
                    logits = tf.get_default_graph().get_tensor_by_name(
                        layer.scope_name + '/cluster/while/Exit_1:0')

                    #get the final squash factor
                    sf = tf.get_default_graph().get_tensor_by_name(
                        layer.scope_name + '/cluster/squash/div_1:0')

                    #get the routing weight
                    #weights = layer.probability_fn(logits)
                    logits = layer.probability_fn(logits)

                    #weights *= tf.transpose(sf, [0, 2, 1])
                    weights = logits*tf.transpose(sf, [0, 2, 1])
                    weights = tf.expand_dims(tf.expand_dims(weights, 1), 4)

                    contrib = tf.reduce_sum(contrib_predict*weights, 2)

            alignment = tf.reduce_sum(
                contrib*tf.expand_dims(capsules, 1), 3,
                name='alignment')
            capsules = tf.identity(capsules, 'output_capsules')

        return capsules, alignment, logits

    def average_capsules(self, output_capsules, targets):
        ''' compute the average capsules
        args:
            output_capsules: [batch_size x num_values x capsule_dim]
            targets: ground truth task labels [batch_size x num_labels]  (numlabels=numoutputcapsules)
        returns:
            average_capsules: [batch_size x capsule_dim]
        '''
        with tf.variable_scope('average_capsules'):

            # output_capsules = tf.identity(output_capsules,'output_capsules')

            # if you dont want to do masking, use the flag and set targets=ones
            flag = None
            if flag == None:
            
            # if youre using masking, use targets:
            # if targets == None:

                # This line for using every output capsule in testing phase
                targets = tf.ones([output_capsules.shape[0],output_capsules.shape[1]], tf.float32)

                # targetlist = []

                # labelprobs = ops.safe_norm(output_capsules)  # first to probs, then make decision

                #for i in range(0, output_capsules.shape[0]):
                #    vector = labelprobs[i, :]
                #    threshold = tf.math.minimum(float(self.conf['dec_threshold']), tf.reduce_max(vector))  # from typesplitcoder.py
                #    vector = tf.where(vector >= threshold, tf.ones_like(vector), tf.zeros_like(vector))
                #    targetlist.append(vector)

                #targets = tf.stack(targetlist)

            # expand dims makes targets from e.g. (2,3) -> (2,3,1)
            # tile copies it, so (2,3,1) with tile (1,1,4) gives (2,3,4) with copies of the (2,3) matrix across 4
            mask = tf.tile(tf.expand_dims(targets, 2), [1, 1, output_capsules.shape[2]])

            masked_output_capsules = output_capsules*mask
            
            avg_capsules = tf.reduce_sum(masked_output_capsules, 1)  ##result = 16x8

            for i in range(0,masked_output_capsules.shape[0]):  # 16 (batch size)
                sum_norms = 0
                for j in range(0, masked_output_capsules.shape[1]):  ## 16x33x8 --> 33
                    a = tf.norm(masked_output_capsules[i, j, :])
                    sum_norms += tf.norm(masked_output_capsules[i, j, :])

                tf.divide(avg_capsules[i, :], sum_norms)

            avg_capsules = tf.identity(avg_capsules, 'average_capsules')
            mask = tf.identity(mask, 'mask')
            masked_output_capsules = tf.identity(masked_output_capsules, 'masked_output')

        return avg_capsules, mask, masked_output_capsules

    def slot_filling(self, output_capsules, alignments):
        '''assign the output capsules to the appropriate slot

        args:
            output_capsules: [batch_size x num_values x capsule_dim]
            alignments: the alignments of the output_capsules
                [batch_size x time x num_values]

        returns:
            the output label probabilities: [batch_size x num_labels]
        '''

        with tf.variable_scope('slot_filling'):

            valids = self.coder.valids
            ids = []
            probs = []
            alis = []
            #all_outputs = tf.concat(tf.unstack(output_capsules, axis=1), 1)

            for i, val in enumerate(valids):
                with tf.variable_scope(val):
                    capsule = output_capsules[:, i]
                    alignment = alignments[:, :, i]
                    alignment = tf.expand_dims(alignment, 2)
                    p = tf.layers.dense(
                        capsule,
                        len(valids[val]),
                        tf.nn.softmax,
                        name=val)
                    alignment *= tf.expand_dims(tf.square(p), 1)
                    p *= ops.safe_norm(capsule, keepdims=True)
                    probs.append(p)

                ids += valids[val].values()
                alis.append(alignment)

            probs = tf.concat(probs, 1)
            probs = tf.gather(probs, ids, axis=1)
            alignments = tf.concat(alis, 2)
            alignments = tf.gather(alignments, ids, axis=2)

        return probs, alignments

    @property
    def num_output_capsules(self):
        '''number of output capsules'''

        if self.conf['slot_filling'] == 'True':
            return len(self.coder.valids)
        else:
            return self.coder.numlabels
