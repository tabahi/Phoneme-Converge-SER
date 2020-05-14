# Speech emotion recognition by phoneme type convergence


Includes [formantfeatures](https://github.com/tabahi/formantfeatures)  — forked from:
<https://github.com/tabahi/formantfeatures>


## SER model training
Training function `PhonemeSER.Train_model()` uses [K-means clustering (scikit-learn)](https://scikit-learn.org/stable/modules/clustering.html#k-means) to label the similar phonological units as the same phonemes. Then phoneme occurrences in whole DB are counted for each clip and occurrence rate is used to train [classifiers](https://scikit-learn.org/stable/modules/svm.html#svm-classification).

Once the model is trained and saved to a file, it can be tested by passing the testing set to `PhonemeSER.Test_model()`

Run complete procedure of training and testing by calling `Train_Test.run_train_and_test` with set parameters:

```python
def run_train_and_test(db_names_paths, results_csv, val_scheme='single-random', test_size=0.2, folds_n=5, test_speakers=5, Nm_inst=[32, 64], Nm_diff=[32, 64], K_SD=0, g_dist=8, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, deselect_labels=None, db_names_paths_2=None):
```

### Parameters

`db_names_paths`: Dict list of names and paths of DBs. Shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], Choose db_name from these: {"EmoDB", "RAVDESS", "IEMOCAP", "ShemoDB", "DEMoS"}
with respective directory paths of the DB wav files. Importing of each DB is programmed in 'Dataset_Lib/SER_DB.py'. The organization format of annotated wav files is expected to be as is downloaded from their original sources. For val_scheme='cross-corpus', all these DBs are used as training sets and `db_names_paths_2` must be passed as testing DB list.

`results_csv`: string, optional (default=None), CSV file path to which final results will be appended.

`val_scheme`: {'L1SO', 'LMSO', 'single-random', 'k-folds', 'cross-corpus'}, optional (default=single-random). Select one of these validation schemes: 'L1SO' : Leave One Speaker Out, 'LMSO': Leave multiple speakers out (default: 5 speakers, require parameter 'test_speakers'), 'single-random': Single random split (requires param 'test_size'), 'k-folds': K-folds cross-validation (require param: 'folds_n').

`test_size`: float, optional (default=0.2). Fraction of total clips to use as testing set for 'single-random' validation scheme. Ignored for other schemes.

`folds_n`: unsigned int, optional (default=5). Number of folds for 'k-folds' validation scheme. Ignored for other schemes.

`test_speakers`: unsigned int, optional (default=5). Number of test speakers for 'LMSO' validation scheme. Ignored for other schemes.

`Nm_inst`: array-like, dtype=int16, shape = [n_models], optional (default=[32, 64]). Cluster numbers for instantaneous (~25ms) phoneme clustering model. Set between 8 and 300 for each cluster model.

`Nm_diff`: array-like, dtype=int16, shape = [n_models], optional (default=[32, 64]). Cluster numbers for differential phoneme (~25ms * g_dist) clustering models. Set between 8 and 300 for each cluster model.

`K_SD`: float, optional (default=0.0). Feature selection parameters. Set between -1 to 1. It sets the limit of standard deviation below the mean for selecting features within this threshold. Lower value selects more features.

`g_dist`: unsigned int, optional (default=8). Number for adjacent frames for measuring the change in formant features to calculate differential phoneme features.

`deselect_labels`: list of chars, optional (default=None). Example: deselect_labels=['F', 'B'] to deselect 'F'and 'B'.

`db_names_paths_2`: Dict list of of test DBs, shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], optional (default=None). Same as 'db_names_paths', but only required for 'cross-corpus' validation scheme.

### Returns

`return 0` : on successful execution.


## Single file test

Once a model is trained and saved to a pickle file, it can be used to predict the label of single WAV file:
```python
test_wav = "test.wav"

import SER_phonemes_learn.PhonemeSER as PhonemeSER
# Make sure an already trained model file is available.
model_file = "data/model_file.pkl"
PhonemeSER.Test_model_wav_file(model_file, test_wav)
'''
Output:
Classifier : Label
SVC1 N
SVC2 N
SVC3 H
SVC4 H
SVC5 A
RF02 A
RF01 H
RF03 H
RF04 H
KNN1 H
MLP1 S
Done
'''
```

------------

## Dependencies


scikit-learn (v2.1), numba (v0.45.1), pickle, matplotlib

