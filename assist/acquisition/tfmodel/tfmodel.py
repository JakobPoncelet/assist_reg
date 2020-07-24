'''@file tfmodel.py
contains the TFModel'''

import os
import shutil
import glob
from operator import mul
from random import shuffle
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from assist.acquisition.model import Model
from assist.tasks.read_task import read_task
from ConfigParser import ConfigParser

class TFModel(Model):
    '''a TensofFlow model'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, coder, expdir):
        '''model constructor

        Args:
            conf: the model configuration as as dictionary of strings
            coder: an object that encodes the tasks
            expdir: the experiments directory
        '''

        super(TFModel, self).__init__(conf, coder, expdir)
        self.is_training = False

    def train(self, examples, all_words, wordfactors):
        '''train the model

        Args:
            examples: the training examples as a dictionary of pairs containing
                the inputs and reference tasks
            all_words: list with all words said by any speaker, ordered alphabetically
        '''

        self.is_training = True

        #create the graph
        graph = tf.Graph()

        for k in examples.keys():
            if (examples[k][0].shape[0]<=5) or not np.isfinite(examples[k][0]).all():
                del examples[k]

        features, tasks = zip(*examples.values())
        voices = examples.keys()

        #encode the speakers
        dataconf = ConfigParser()
        dataconf.read(os.path.join(self.expdir, 'database.cfg'))
        nr_spk = len(dataconf.sections())
        print('nr_spk: ',nr_spk)
        speakers_ordered = dataconf.sections()
        print(speakers_ordered)

        encoded_spk = np.zeros(len(tasks)) # one-hot encoding of speakers
        cnt = 0
        for voice in voices:
            spk_id = '_'.join(voice.split('_')[:1])
            encoded_spk[cnt] = speakers_ordered.index(spk_id)
            cnt += 1

        #encode all the words
        encoded_words = np.zeros((len(all_words), len(tasks))) 
        voicestemp = list(voices)
        for speaker in speakers_ordered:
            result = sorted([x for x in voicestemp if x.startswith(speaker+'_')])
            voicestemp = [x for x in voicestemp if x not in result]

            if result:  #check if not empty
                taskloc = dataconf.get(speaker, 'tasks')  # to get folder of database with the text written out
                textloc = taskloc[:-5] + str('text')

                with open(textloc) as fp:
                    line = (fp.readline())[:-1]

                    while line:
                        voice = speaker + '_' + str((line.split(" "))[0])
                        sentence = (line.split(" "))[1:]

                        if voice in result:
                            corr_index = voices.index(voice)

                            for word in sentence:
                                pos = all_words.index(word)
                                encoded_words[pos, corr_index] = 1

                        line = (fp.readline())[:-1]
        encoded_words = np.transpose(encoded_words)

        #read all the tasks
        tasks = [read_task(task) for task in tasks]

        #encode the tasks
        vs = np.array([self.coder.encode(t) for t in tasks])

        if self.conf['batch_size'] == 'None':
            batch_size = features.shape[0]
        else:
            batch_size = min(int(self.conf['batch_size']), len(features))

        with graph.as_default():
            #put the features in a constant
            inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, None, features[0].shape[-1]],
                name='inputs')
            seq_length = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size],
                name='seq_length')

            #put the targets in a constant
            targets = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, vs.shape[-1]],
                name='targets')

            #put the labels in a constant
            speakers = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size],
                name='speakers')

            correct_words = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, len(all_words)],
                name='correct_words')

            expanded_wordfactors_present = np.zeros((batch_size,len(all_words)),np.float32)
            expanded_wordfactors_absent = np.zeros((batch_size,len(all_words)),np.float32)
            for i in range(0,batch_size):
                expanded_wordfactors_present[i,:] = wordfactors[0,:]
                expanded_wordfactors_absent[i,:] = wordfactors[1,:]

            expanded_wordfactors_present = tf.convert_to_tensor(expanded_wordfactors_present,dtype=tf.float32)
            expanded_wordfactors_absent = tf.convert_to_tensor(expanded_wordfactors_absent,dtype=tf.float32)

            #apply the model
            labelprobs, spklogits, wordlogits = self.model(inputs, seq_length, nr_spk, all_words, targets)
            loss, labelloss, spkloss, wordloss = self.loss(targets, speakers, correct_words, labelprobs, spklogits, wordlogits, expanded_wordfactors_present, expanded_wordfactors_absent)

            #count the number of parameters
            num_params = 0
            for var in tf.trainable_variables():
                num_params += reduce(mul, var.get_shape().as_list())
            print 'number of parameters: %d' % num_params

            #create an optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=float(self.conf['learning_rate']), epsilon=1e-03)

            #compute the gradients
            grads_and_vars = optimizer.compute_gradients(loss=loss)

            with tf.variable_scope('clip'):
                #clip the gradients
                grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var)
                                  for grad, var in grads_and_vars]

            #opperation to apply the gradients
            apply_gradients_op = optimizer.apply_gradients(
                grads_and_vars=grads_and_vars,
                name='apply_gradients')

            #all remaining operations with the UPDATE_OPS GraphKeys
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            #create an operation to update the model
            update_op = tf.group(
                *([apply_gradients_op] + update_ops),
                name='update')

            #create the init op
            init_op = tf.variables_initializer(tf.global_variables())

            #create a saver
            saver = tf.train.Saver()

            #create a summary
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            tf.summary.scalar('loss', loss)

            if self.conf['images'] == 'True':
                images = tf.get_collection('image')
                for image in images:
                    tf.summary.image(image.name, image)

            summary = tf.summary.merge_all()

        #create a session
        session_conf = tf.ConfigProto(
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0
        )
        sess = tf.Session(graph=graph, config=session_conf)

        #create a summary writer
        writer = tf.summary.FileWriter(
            os.path.join(self.expdir, 'logdir'),
            graph)

        #initialize the model
        sess.run(init_op)

        #create an index queue
        index_queue = []
        for _ in range(int(self.conf['numiters'])):
            i = range(len(tasks))
            shuffle(i)
            index_queue += i

        #iterativaly train the model
        i = 0
        while len(index_queue) > batch_size:
            indices = index_queue[:batch_size]
            index_queue = index_queue[batch_size:]
            batch_inputs = [features[j] for j in indices]
            batch_lengths = np.array([f.shape[0] for f in batch_inputs])
            ml = np.max(batch_lengths)
            batch_inputs = np.array(
                [np.pad(f, ((0, ml-f.shape[0]), (0, 0)), 'constant')
                 for f in batch_inputs]
            )
            if i == 'a':
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = run_metadata = None
            _, s, l, lbl, spkl, spklgts, wrdl, wrdlgts = sess.run(
                (update_op, summary, loss, labelloss, spkloss, spklogits, wordloss, wordlogits),
                feed_dict={
                    inputs: batch_inputs,
                    seq_length: batch_lengths,
                    targets: vs[indices],
                    speakers: encoded_spk[indices],
                    correct_words: encoded_words[indices]
                },
                options=run_options,
                run_metadata=run_metadata)
            print 'step %d: loss = %f' % (i, l)
            print '     labelloss = %f' %lbl
            print '     spkloss = %f' %spkl
            print '     wordloss = %f' %wrdl
            writer.add_summary(s, i)
            if i == 'a':
                writer.add_run_metadata(run_metadata, 'statistics')
            i += 1

        #save the final model
        saver.save(sess, os.path.join(self.expdir, 'logdir', 'model.ckpt'))

        sess.close()

        self.is_training = False

    def decode(self, inputs, all_words, word_thresholds): ####, all_targets):
        '''decode using the model

        Args:
            inputs: the inputs as a dictionary

        Returns:
            the estimated task representations as a dictionary
        '''

        #create the graph
        graph = tf.Graph()

        dataconf = ConfigParser()
        dataconf.read(os.path.join(self.expdir, 'database.cfg'))
        nr_spk = len(dataconf.sections())
        print('nr_spk: ',nr_spk)
        speakers_ordered = dataconf.sections()
        print(speakers_ordered)

        #pad the features to equal length
        lengths = np.array([inp.shape[0] for inp in inputs.values()])
        ml = np.max(lengths)
        features = np.array([
            np.lib.pad(f, ((0, ml-f.shape[0]), (0, 0)), 'constant')
            for f in inputs.values()])
        dummy = np.zeros_like(features[0])

        batch_size = int(self.conf['batch_size'])

        wordthresholds = [word_thresholds]
        for i in range(0,batch_size-1):
            wordthresholds.append(word_thresholds)

        with graph.as_default():
            #put the features in a constant
            inp = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, ml, features.shape[-1]],
                name='inputs')
            seq_length = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size],
                name='seq_length')

            targets = None
            ####targets = tf.placeholder(
            ####    dtype=tf.float32,
            ####    shape=[batch_size, 33],
            ####    name='targets')

            #apply the model
            labelprobs, spklogits, wordlogits = self.model(inp, seq_length, nr_spk, all_words, targets)
            spkprobs = tf.nn.softmax(spklogits, 1)
            wordprobs = tf.nn.sigmoid(wordlogits)
            if not word_thresholds:
                thresholded_words = tf.to_int32(wordprobs > float(self.conf['word_threshold']))  #set all words with prob > threshold to 1, otherwise 0
            else:
                thresholds = tf.convert_to_tensor(wordthresholds,dtype=tf.float32)
                comparison = tf.greater_equal(wordprobs,thresholds)  #boolean tensor with element-wise True if x>=y
                thresholded_words = tf.cast(comparison, tf.int32)  #cast True/False to 1/0

            #create a saver
            saver = tf.train.Saver()

            #get the images
            if self.conf['images'] == 'True':
                images = tf.get_collection('image')
                for image in images:
                    tf.summary.image(image.name.split('/')[-1], image)

                #get the data to store
                to_store = tf.get_collection('store')
                to_store = {d.op.name.split('/')[-1]: d for d in to_store}
            else:
                to_store = {}

            summary = tf.summary.merge_all()
            if summary is None:
                summary = []

        #create a session
        sess = tf.Session(graph=graph)

        #create a summary writer
        writer = tf.summary.FileWriter(
            os.path.join(self.expdir, 'logdir-decode'),
            graph)

        for v in to_store:
            if os.path.isdir(os.path.join(self.expdir, v)):
                shutil.rmtree(os.path.join(self.expdir, v))
            os.makedirs(os.path.join(self.expdir, v))

        #load the model
        saver.restore(sess, os.path.join(self.expdir, 'logdir', 'model.ckpt'))

        outputs = {}
        decoded_speakers = {}
        decoded_words = {}

        for i in range(int(len(features)/batch_size)):
            batch_inputs = features[i*batch_size:(i+1)*batch_size]  # array
            batch_lengths = lengths[i*batch_size:(i+1)*batch_size]  # array
            batch_names = inputs.keys()[i*batch_size:(i+1)*batch_size]  
            ####batch_targets = np.array([all_targets[utt] for utt in batch_names])
            o, q, w, s, d = sess.run(
                (labelprobs, spkprobs, thresholded_words, summary, to_store),
                feed_dict={inp:batch_inputs, seq_length:batch_lengths}) ####, targets:batch_targets})
            for j, name in enumerate(batch_names):
                dec_spk = speakers_ordered[np.argmax(q[j, :])]
                decoded_speakers[name] = dec_spk
                decoded_words[name] = w[j,:]
                outputs[name] = o[j]

                for v, t in d.items():
                    np.save(os.path.join(self.expdir, v, '%s.npy' % name), t[j])
            if s:
                writer.add_summary(s, i)
        rem = features.shape[0] % batch_size
        if rem > 0:
            batch_inputs = np.concatenate(
                [features[-rem:], np.array([dummy]*(batch_size-rem))])
            batch_lengths = np.concatenate(
                [lengths[-rem:], np.array([0]*(batch_size-rem))])
            batch_names = inputs.keys()[-rem:]
            ####batch_targets = np.concatenate([[all_targets[utt] for utt in batch_names], np.zeros([batch_size-rem,33])])
            o, q, w, s, d = sess.run(
                (labelprobs, spkprobs, thresholded_words, summary, to_store),
                feed_dict={inp:batch_inputs, seq_length:batch_lengths}) ####, targets:batch_targets})
            for j, name in enumerate(batch_names):
                dec_spk = speakers_ordered[np.argmax(q[j, :])]
                decoded_speakers[name] = dec_spk
                decoded_words[name] = w[j,:]
                outputs[name] = o[j]

                for v, t in d.items():
                    np.save(os.path.join(self.expdir, v, '%s.npy' % name), t[j])
            if s:
                writer.add_summary(s, int(len(features)/batch_size))

        sess.close()

        #get the task representation
        tasks = {o: self.coder.decode(outputs[o], _sig_cross_entropy)
                 for o in outputs}

        return tasks, decoded_speakers, decoded_words

    @abstractmethod
    def model(self, inputs, seq_length, nr_spk, all_words, targets):
        '''
        apply the model

        args:
            inputs: the model inputs [batch_size x time x input_dim]
            seq_length: the input sequence length [batch_size]
            nr_spk: number of speakers in database
            all_words: list with all words said by any speaker, ordered alphabetically
            targets: ground truth labels during training phase (for masking the output capsules in speakerpart)

        returns: the label probabilities [batch_size x num_labels]
        '''

    @abstractmethod
    def loss(self, targets, speakers, correct_words, labelprobs, spklogits, wordlogits, wordfactors_present, wordfactors_absent):
        '''compute the loss

        args:
            targets: the reference targets
            labelprobs: the label probabilities
            spkprobs: the speaker probabilities
        returns: the loss'''

        pass

    def load(self, directory):
        '''load the model

        Args:
            directory: the directory where the model was saved
        '''

        #create the logdir if it does not exist
        if not os.path.isdir(os.path.join(self.expdir, 'logdir')):
            os.makedirs(os.path.join(self.expdir, 'logdir'))

        #create the checkpoint file
        with open(
            os.path.join(self.expdir, 'logdir', 'checkpoint'), 'w') as fid:

            fid.write('model_checkpoint_path: "%s"' % (
                os.path.join(self.expdir, 'logdir', 'model.ckpt')))
            fid.write('all_model_checkpoint_paths: "%s"' % (
                os.path.join(self.expdir, 'logdir', 'model.ckpt')))

        #copy the model files
        for f in glob.glob('%s*' % os.path.join(directory, 'model.ckpt')):
            shutil.copy(f, os.path.join(self.expdir, 'logdir'))

    def save(self, directory):
        '''save the model

        Args:
            directory: the directory where the model should be saved
        '''

        #create the directory if it does not exist
        if not os.path.isdir(directory):
            os.makedirs(directory)

        #create the checkpoint file
        with open(os.path.join(directory, 'checkpoint'), 'w') as fid:
            fid.write('model_checkpoint_path: "%s"' % (
                os.path.join(directory, 'model.ckpt')))
            fid.write('all_model_checkpoint_paths: "%s"' % (
                os.path.join(directory, 'model.ckpt')))

        #copy the model files
        for f in glob.glob(
                '%s*' % os.path.join(self.expdir, 'logdir', 'model.ckpt')):
            shutil.copy(f, directory)

	for f in glob.glob(
		'%s*' % os.path.join(self.expdir,'logdir')):
	    shutil.rmtree(f,ignore_errors = True)

def _sig_cross_entropy(v, x):
    '''compute the cross-entropy between v and x'''

    #clip x to avoid nans
    x = np.clip(x, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    return -(v*np.log(x) + (1-v)*np.log(1-x)).sum()

