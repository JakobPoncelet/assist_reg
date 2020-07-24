'''@file train_test.py
do training followed by testing
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import numpy as np
import glob
import shutil
import cPickle as pickle
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.tasks import read_task
from assist.acquisition import model_factory
from assist.experiment import score

def main(expdir):
    '''main function'''

    #check if this experiment has been completed
    if os.path.exists(os.path.join(expdir, 'f1')):
        print 'result found %s' % expdir
        return

    #read the acquisition config file
    acquisitionconf = ConfigParser()
    acquisitionconf.read(os.path.join(expdir, 'acquisition.cfg'))

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    #create a task structure file
    structure = Structure(os.path.join(expdir, 'structure.xml'))

    #create a coder
    coder = coder_factory.factory(coderconf.get('coder', 'name'))(
        structure, coderconf)

    #create an acquisition model
    model = model_factory.factory(acquisitionconf.get('acquisition', 'name'))(
        acquisitionconf, coder, expdir)

    print 'loading model'
    model.load(os.path.join(expdir, 'model'))

    print 'prepping testing data'

    #load the testing features
    features = dict()
    for line in open(os.path.join(expdir, 'testfeats')):
        splitline = line.strip().split(' ')
        featsfile = ' '.join(splitline[1:])
        features[splitline[0]] = np.load(featsfile)

    #read the testtasks
    references = dict()
    tasklabels = dict()
    for line in open(os.path.join(expdir, 'testtasks')):
        splitline = line.strip().split(' ')
        taskstring = read_task.read_task(' '.join(splitline[1:]))
        references[splitline[0]] = taskstring
        tasklabels[splitline[0]] = coder.encode(taskstring)
		
    #find all words said by speakers and save all spoken sentences in testtasks
    wordcount = {}
    testsentences = {}
    dataconf = ConfigParser()
    dataconf.read(os.path.join(expdir, 'database.cfg'))
    for speaker in dataconf.sections():
        taskloc = dataconf.get(speaker, 'tasks')
        textloc = taskloc[:-5]+str('text')
        with open(textloc) as fp:
            line = (fp.readline())[:-1]
            while line:
                sentence = (line.split(" "))[1:]
                voice = str(speaker)+'_'+(line.split(" "))[0]
                for word in sentence:
                    if word in wordcount:
                        wordcount[word] += 1
                    else:
                        wordcount[word] = 1
                if voice in references:
                    testsentences[voice] = " ".join(sentence)
                line = (fp.readline())[:-1]
    all_words = sorted(wordcount.keys())  #ordered alphabetically
	
    #read the singlebest word for each label
    singlebest = []
    with open(os.path.join(expdir, 'singlebestwords'),'r') as fp:
        line = (fp.readline())[:-1]
        while line:
            word = (line.split(" "))[1]
            singlebest.append(word)
	    line = (fp.readline())[:-1]

    #get the singlebest predictions for each voicing		
    singlebest_sentences = {}
    for voice in tasklabels:
        label = tasklabels[voice]
	indices = np.nonzero(label)
	sentence = ''
        print(indices)
	for i in indices[0]:
	    word = singlebest[i]
	    sentence = sentence + ' ' + str(word)
	singlebest_sentences[voice] = sentence
	
    with open(os.path.join(expdir,'singlebestsentences'),'w') as fid:
        for name, sentence in sorted(singlebest_sentences.items()):
            fid.write('%s %s\n' % (name,sentence))
	
    print 'testing the model'

    word_thresholds = None
    # if you would want word specific thresholds ...
    threshfile = os.path.join(expdir, 'word_thresholds.pkl')
    if os.path.isfile(threshfile):
        with open(threshfile, 'r') as fid:
            word_dict = pickle.load(fid)
        word_thresholds = []
        for word, thresh in sorted(word_dict.items()):
            word_thresholds.append(thresh)

    print 'LAUNCHING MODEL.DECODE'
    decoded_speakers = dict()
    decoded_words = dict()

    # filter out features that are too short 
    # (+ corresponding references to not get errors in score.score)
    for k in features.keys():
        if features[k].shape[0] <= 5:
            del features[k]
            del references[k]

    #decode the test uterances
    if acquisitionconf.get('acquisition', 'name') == 'nmf':
        decoded = model.decode(features)
    else:
    #### decoded, decoded_speakers, decoded_words = model.decode(features, all_words, word_thresholds, tasklabels)
        decoded, decoded_speakers, decoded_words = model.decode(features, all_words, word_thresholds)
    #decoded = model.decode(features)


    #write the decoded tasks to disc
    with open(os.path.join(expdir, 'dectasks'), 'w') as fid:
        for name, task in decoded.items():
            fid.write('%s %s\n' % (name, read_task.to_string(task)))

    if acquisitionconf.get('acquisition', 'name') != 'nmf':
        with open(os.path.join(expdir, 'decspeakers'), 'w') as fid:
            for name, spk in decoded_speakers.items():
                fid.write('%s %s\n' % (name, spk))

        speakerperformance = score.spk_score(decoded_speakers)
        print 'speakerperformance: %f' % speakerperformance

        with open(os.path.join(expdir, 'speakerperformance'), 'w') as fid:
            fid.write(str(speakerperformance))

        decoded_sentences = {}
        with open(os.path.join(expdir, 'decwords'), 'w') as fid:
            for name, wordslist in sorted(decoded_words.items()):
                sentence = ''
                wordindices = np.nonzero(wordslist)  #returns all indices that are nonzero
                for wordindex in wordindices[0]:
                    word = all_words[wordindex]
                    sentence = sentence + ' ' + str(word)
                fid.write('%s %s\n' % (name, sentence))
                decoded_sentences[name] = sentence

        with open(os.path.join(expdir, 'testwords'),'w') as fid:
            for name, sentence in sorted(testsentences.items()):
                fid.write('%s %s\n' % (name, sentence))
	
        word_f1, word_precision, word_recal = score.wordscore(decoded_sentences, testsentences)
	
        singlebest_f1, singlebest_precision, singlebest_recal = score.wordscore(singlebest_sentences, testsentences)
	
        print 'word_f1: %f' % word_f1
        print 'word_precision: %f' % word_precision
        print 'word_recal: %f' % word_recal
        with open(os.path.join(expdir, 'word_f1'), 'w') as fid:
            fid.write(str(word_f1))
        with open(os.path.join(expdir, 'word_precision'), 'w') as fid:
            fid.write(str(word_precision))
        with open(os.path.join(expdir, 'word_recal'), 'w') as fid:
            fid.write(str(word_recal))
		
        print 'singlebest_f1: %f' % singlebest_f1
        print 'singlebest_precision: %f' % singlebest_precision
        print 'singlebest_recal: %f' % singlebest_recal
        with open(os.path.join(expdir, 'singlebest_f1'), 'w') as fid:
            fid.write(str(singlebest_f1))
        with open(os.path.join(expdir, 'singlebest_precision'), 'w') as fid:
            fid.write(str(singlebest_precision))
        with open(os.path.join(expdir, 'singlebest_recal'), 'w') as fid:
            fid.write(str(singlebest_recal))
			
    (precision, recal, f1, macroprec, macrorecall, macrof1), scores = \
        score.score(decoded, references)

    fluent_accuracy = score.fluentscore(decoded, references)
    print 'fluent_accuracy: %f' % fluent_accuracy
    with open(os.path.join(expdir, 'fluent_accuracy'), 'w') as fid:
        fid.write(str(fluent_accuracy))
	
    print 'precision: %f' % precision
    print 'recal: %f' % recal
    print 'f1: %f' % f1
    print 'macro precision: %f' % macroprec
    print 'macro recal: %f' % macrorecall
    print 'macro f1: %f' % macrof1

    with open(os.path.join(expdir, 'precision'), 'w') as fid:
        fid.write(str(precision))
    with open(os.path.join(expdir, 'recal'), 'w') as fid:
        fid.write(str(recal))
    with open(os.path.join(expdir, 'f1'), 'w') as fid:
        fid.write(str(f1))
    with open(os.path.join(expdir, 'macroprecision'), 'w') as fid:
        fid.write(str(macroprec))
    with open(os.path.join(expdir, 'macrorecal'), 'w') as fid:
        fid.write(str(macrorecall))
    with open(os.path.join(expdir, 'macrof1'), 'w') as fid:
        fid.write(str(macrof1))

    score.write_scores(scores, expdir)

    for f in glob.glob(
            '%s*' % os.path.join(expdir,'logdir')):
        shutil.rmtree(f,ignore_errors = True)

    for f in glob.glob(
            '%s*' % os.path.join(expdir,'logdir-decode')):
        shutil.rmtree(f,ignore_errors = True)
	
if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
