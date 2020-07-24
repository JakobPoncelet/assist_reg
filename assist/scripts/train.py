'''@file train_test.py
do training followed by testing
'''
from __future__ import division
import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.acquisition import model_factory
from assist.tasks.read_task import read_task

def main(expdir):
    '''main function'''

    #check if this experiment has been completed
    if os.path.isdir(os.path.join(expdir, 'model')):
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

    print 'prepping training data'

    #load the training features
    features = dict()
    for line in open(os.path.join(expdir, 'trainfeats')):
        splitline = line.strip().split(' ')
        featsfile = ' '.join(splitline[1:])
        features[splitline[0]] = np.load(featsfile)

    #read the traintasks
    taskstrings = dict()
    for line in open(os.path.join(expdir, 'traintasks')):
        splitline = line.strip().split(' ')
        taskstrings[splitline[0]] = ' '.join(splitline[1:])

    task = read_task(taskstrings[splitline[0]])
    label = coder.encode(task)
    num_labels = len(label)
    print('num_labels: ', num_labels)		
    

    #find all words said by speakers and count number of occurrences
    wordcount = {}
    wordcount_in_traintasks = {}
    sentencecount = 0
    sentencecount_in_traintasks = 0
	
    #save file with transcription of traintasks
    trainsentences = {}
    unique_sentencecount = {}
	
    dataconf = ConfigParser()
    dataconf.read(os.path.join(expdir, 'database.cfg'))
    for speaker in dataconf.sections():
        taskloc = dataconf.get(speaker, 'tasks')
        textloc = taskloc[:-5]+str('text')
        with open(textloc) as fp:
            line = (fp.readline())[:-1]
            while line:
                sentencecount += 1
                voice = str(speaker)+'_'+(line.split(" "))[0]
                sentence = (line.split(" "))[1:]
                for word in sentence:
                    if word in wordcount:
                        wordcount[word] += 1
                    else:
                        wordcount[word] = 1
                if voice in taskstrings:
		    trainsentences[voice] = " ".join(sentence)
                    sentencecount_in_traintasks += 1
                    for word in sentence:
                        if word in wordcount_in_traintasks:
                            wordcount_in_traintasks[word] += 1
                        else:
                            wordcount_in_traintasks[word] = 1
		    uniquewords = list(dict.fromkeys(sentence))
		    for word in uniquewords:
		        if word in unique_sentencecount:
			    unique_sentencecount[word] += 1
		        else:
			    unique_sentencecount[word] = 1
                line = (fp.readline())[:-1]
    all_words = sorted(wordcount.keys())  #ordered alphabetically
	
    wordfactors = np.zeros((2,len(all_words)),dtype=np.float32)
    for i in range(0,len(all_words)):
        word = all_words[i]
        if word not in wordcount_in_traintasks:
            continue
        count = wordcount_in_traintasks[word]
        if count == sentencecount_in_traintasks:
            count -= 1
        wordfactors[0,i] = sentencecount_in_traintasks/(2*count)  #factor for present
        wordfactors[1,i] = sentencecount_in_traintasks/(2*(sentencecount_in_traintasks-count))  #factor for absent
        print('Word: ',word,'     Wordfactors (present vs absent): ',wordfactors[0,i],' and ',wordfactors[1,i])
    
    #create lists of features and training tasks
    examples = {utt: (features[utt], taskstrings[utt]) for utt in taskstrings}
	
    #calculate the TF
    #for every tasklabel, count frequency of words in trainsentences corresponding to each tasklabel
    wordfreq_matrix = np.zeros((num_labels,len(all_words)))
    for voice in taskstrings:
        taskstring = taskstrings[voice]
        task = read_task(taskstring)
        labels = coder.encode(task)
        sentence = trainsentences[voice]
        for word in sentence.split(" "):
            wordind = all_words.index(word)
            wordfreq_matrix[:,wordind] += labels  # term frequency TF
	
    #calculate the IDF
    N = sentencecount_in_traintasks  # total number of sentences in the trainingset
    D = np.zeros(len(all_words))  # vector with for each word in how many sentences in trainingset the word occurs
    for word in unique_sentencecount:
        wordind = all_words.index(word)
        count = unique_sentencecount[word]
        D[wordind] = count
    IDF = np.array([np.log(N/(1+x)) for x in D])  # inverse document frequency
	
    TFIDF = wordfreq_matrix * IDF  # TFIDF matrix
	
    print(coder.argindices.items())  # get all labels and corresponding capsule number

    #save the single best chosen to a file
    with open(os.path.join(expdir, 'singlebestwords'), 'w') as fid:
        for i in range(0, num_labels):
            singlebest_ind = np.argmax(TFIDF[i,:])
            singlebest = all_words[singlebest_ind]
            fid.write('%i %s\n' % (i, singlebest))  # hier nog labelnaam in krijgen
			
    #save the trainsentences to a file
    with open(os.path.join(expdir, 'trainwords'),'w') as fid:
        for name, sentence in sorted(trainsentences.items()):
            fid.write('%s %s\n' % (name, sentence))
    
    if acquisitionconf.get('acquisition', 'name') == 'nmf':
        model.train(examples)
    else:
        print 'training acquisition model (LAUNCHING MODEL.TRAIN)'
        model.train(examples, all_words, wordfactors)
    
    #model.train(examples)

    #save the trained model
    model.save(os.path.join(expdir, 'model'))

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
