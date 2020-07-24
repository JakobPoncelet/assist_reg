'''@file compare_results.py
compare the results of experiments on the same database

usage: python compare_results.py result expdir1, expdir2, ...
    expdir: the experiments directory of one of the experiments
    result: what you want to plot (e.g. f1)

- smoothing can be changed from smooth1 to smooth2
(spkindep vs spkdep plots)

- set labelnames in script
'''

import sys
import os
import itertools
import numpy as np
import argparse
from ConfigParser import ConfigParser
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from get_results import get_results

def main(expdirs, result):
    '''main function'''

    ## delete the line matplotlib.use('Agg') if you want to show the plots
    toplot = result
    resultdir = expdirs
    expdirs = [os.path.normpath(expdir) for expdir in expdirs]

    #colorlist = ['red', 'blue', 'cyan', 'green', 'yellow', 'magenta',
    #             'purple', 'pink', 'gold', 'navy', 'olive', 'grey']
    #linestyles = ['-']

    colorlist = ['black']
    linestyles = ['-', '--', ':', '-.',(0, (5, 10))]

    #colorlist = ['green', 'dimgrey', 'darkorange']
    #linestyles = ['-']

    plot_speakers = False  # True
    remove_uncomplete = True  #False

    #tick parameters
    tick_params = {
        'size': 'x-large',
        #'color': 'dimgrey'
    }

    #axis properties
    ax_params = {
        'color':'black'
    }

    #label properties
    label_params = {
        'color':'black',
        'size': 'x-large'
    }

    #legend properties
    legend_params = {
        'loc': 'lower right',
        'edgecolor':'black',
        'fontsize': 'x-large'
    }
    lcolor = 'black'

    #lowess parameters
    def smooth1(y,x):
        return lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=2.0/3,
        it=0,
        delta=1.0,
        return_sorted=True)

    #weighted moving average
    def smooth2(y,x,step_size=0.1,width=50):
        bin_centers  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
        bin_avg = np.zeros(len(bin_centers))

        #weight with a Gaussian function
        def gaussian(x,amp=1,mean=0,sigma=1):
            return amp*np.exp(-(x-mean)**2/(2*sigma**2))

        for index in range(0,len(bin_centers)):
            bin_center = bin_centers[index]
            weights = gaussian(x,mean=bin_center,sigma=width)
            bin_avg[index] = np.average(y,weights=weights)

        yvals = np.sort(bin_avg)
        xvals = [bin_centers[i] for i in list(np.argsort(bin_avg))]
        return np.transpose(np.array([xvals, yvals]))

    #read all the results
    results = [get_results(expdir, result) for expdir in expdirs]
    expnames = [os.path.basename(expdir) for expdir in expdirs]
    
    labelnames = ['pccn', 'rccn', 'nmf', 'encoder-decoder']
    #labelnames = ['pccn-multi', 'rccn-multi', 'pccn', 'rccn']

    smooth = smooth2

    pickylabel = 'f1'
    if toplot == 'word_f1':
        pickylabel = 'word_f1'
    elif toplot == 'speakerperformance':
        pickylabel = '% correctly decoded speakers' 

    spkweights = []
    wordweights = []
    wordthresholds = []

    #remove experiments that are not performed in all experiments
    if remove_uncomplete:
        speakers = set(results[0].keys())
        for result in results[1:]:
            speakers = speakers & set(result.keys())
        results = [{s: result[s] for s in speakers} for result in results]
        for speaker in speakers:
            experiments = set(results[0][speaker].keys())
            for result in results[1:]:
                experiments = experiments & set(result[speaker].keys())
            if not experiments:
                for result in results:
                    del result[speaker]
            else:
                for result in results:
                    result[speaker] = {
                        e: result[speaker][e] for e in experiments}

    wordweights = [1, 10, 100]
    if plot_speakers:
        for speaker in results[0]:
            plt.figure(speaker)
            for i, result in enumerate(results):
                if speaker not in result:
                    continue
                sort = np.array(result[speaker].values())
                sort = sort[np.argsort(sort[:, 0], axis=0), :]
                fit = smooth(sort[:, 1], sort[:, 0])
                plot = plt.plot(
                    fit[:, 0], fit[:, 1],
                    color=colorlist[i%len(colorlist)],
                    linestyle=linestyles[i%len(linestyles)])
                   # label=expnames[i])
            plt.yticks(**tick_params)
            plt.xticks(**tick_params)
            plt.axis(**ax_params)
            l = plt.legend(**legend_params)
            for text in l.get_texts():
                text.set_color(lcolor)
            plt.xlabel('# Trainingsvoorbeelden', **label_params)
            plt.ylabel('Accuracy', **label_params)

    #concatenate all the results
    concatenated = [
        np.array(list(itertools.chain.from_iterable(
            [r.values() for r in result.values()])))
        for result in results]
    #sort the concatenated data
    sort = [c[np.argsort(c[:, 0], axis=0), :] for c in concatenated]
    #smooth all the results
    fit = [smooth(s[:, 1], s[:, 0]) for s in sort]
    plt.figure('result '+str(toplot))
    for i, f in enumerate(fit):
        plt.plot(f[:, 0], f[:, 1],
                 color=colorlist[i%len(colorlist)],
                 linestyle=linestyles[i%len(linestyles)],
                 label=labelnames[i])

    plt.yticks(**tick_params)
    plt.xticks(**tick_params)
    plt.axis(**ax_params)
    l = plt.legend(**legend_params)
    for text in l.get_texts():
        text.set_color(lcolor)
    plt.xlabel('# Examples', **label_params)
    plt.ylabel(str(pickylabel), **label_params)
    print 'Figure of result saved in: ',resultdir[0]
    plt.savefig(os.path.join(resultdir[0],'result_'+str(toplot)+'.pdf'),format='pdf')
    plt.show()


if __name__ == '__main__':

    main(sys.argv[2:], sys.argv[1])
