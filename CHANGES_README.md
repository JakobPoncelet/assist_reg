# ABOUT RUNNING THE CODE

1) The scripts use speaker recognition and word reconstruction by default.
To disable:
 - set wordweight and speakerweight to 0 in the configuration file
 - set trainable=False in tfmodel/reconstruction.py and tfmodel/speakerprobabilities.py
 - comment wordloss and speakerloss out of the total loss in rccn_spk.py / pccn_spk.py

2) Speaker dependent experiments can be created with "python ./run cross_validation <args>"
   Speaker independent experiments can be created with "python ./run cross_validation_ppall <args>"


# DETAILED CHANGES

Implemented some changes to the original code // Used on GRABO and FLUENTSPEECHCOMMANDS datasets.

(only if you are familiar with the original code and the original readme files on usage)

In general:
Extended the rccn model with speaker recognition and reconstruction of the words said (words based on the textual transcription).
A new model is made: RCCN_SPK that implements these (in FluentSpeech dataset the config is called RCCN_vad).
All the prepare scripts now also put the speaker identity in the name of the task and make some lists with all words etcetera.
Reuse of saved test/trainids and blocks partitions because it takes a very long time with larger datasets.

The folder 'analysis' contains some files I used for analysis of results, probably not useful to you, and some tips on running.


THE MODEL (rccn_spk is added to the model factory in assist/acquisition)
assist/acquisition/tfmodel/rccn_spk
	- contains the extended model
	- the loss function is also extended to train speakers and words

assist/acquisition/tfmodel/reconstruction
	- new NN tf layer for implementing the words reconstruction

assist/acquisition/tfmodel/speakerprobabilities
	- new NN tf layer for implementing speaker identification

assist/acquisition/tfmodel/tfmodel
	- extended with encoding of all speakers, all words, ...
	- many more things are calculated in the model

INITIALIZATION
assist/scripts/prepare_cross_validation_ppall
	- is used and shuffles all speakers beforehand
	- loads blocks-file and training/test division beforehand from recipe because takes a very long time to make
	  These are called the saved_ids folder and the blocks.pkl and 150blocks.pkl files.
	- if used for first time, the program makes and saves these
	
assist/scripts/train and ./test
	- extended with count of words etc.
	- also scores for word and speaker performance (see addition in assist/experiment/score)
	 
assist/condor
	- for me, only run_script_GPU worked (using cpu's did not work)

PLOTTING
assist/scripts/compare_results
	- change here what you want to plot, which titles to use etc.
