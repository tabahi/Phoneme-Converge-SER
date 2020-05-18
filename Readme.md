# Speech emotion recognition by phoneme type convergence


Includes [formantfeatures](https://github.com/tabahi/formantfeatures)  — forked from:
<https://github.com/tabahi/formantfeatures>


## SER model training
Training function `PhonemeSER.Train_model()` uses [K-means clustering (scikit-learn)](https://scikit-learn.org/stable/modules/clustering.html#k-means) to label the similar phonological units as the same phonemes. Then phoneme occurrences in whole DB are counted for each clip and occurrence rate is used to train [classifiers](https://scikit-learn.org/stable/modules/svm.html#svm-classification).

Once the model is trained and saved to a file, it can be tested by passing the testing set to `PhonemeSER.Test_model()`



------------

## Dependencies

+ Python 3.7 or later
+ Numpy 1.16 or later
+ [Scipy v1.3.1](https://scipy.org/install.html)
+ [H5py v2.9.0](https://pypi.org/project/h5py/)
+ [Scikit-learn (v0.22.1)](https://scikit-learn.org/stable/index.html)
+ [Numba (v0.45.1)](https://numba.pydata.org/numba-doc/dev/user/installing.html)
+ [Wavio v0.0.4](https://pypi.org/project/wavio/)

> Install all: `pip install numpy scipy h5py numba wavio scikit-learn`

-------

[See SER_Datasets_Import module for reading database directory in their relevant filename formants.](https://github.com/tabahi/SER_Datasets_Import)
 Extract formant features and save to an HDF database by passing the DB name and path to `Extract_files_formant_features`:

```python
list_of_clips = SER_Datasets_Libs.SER_DB.create_DB_file_objects("DB_NAME", "C:/DB/path")

 processed_n = FormantsLib.FormantsExtract.Extract_files_formant_features(list_of_clips, "formants_filename.hdf", window_length=0.025, window_step=0.01, f0_min=30, f0_max=4000, max_frames=800, formants=3)
```



Once formant features are extracted and saved to HDF they can be imported again to train the model:

```python
features, labels, u_speakers, u_classes  = HDFread.import_mutiple_HDFs("formants_filename.hdf", deselect_labels=['C', 'D', 'F', 'U', 'E', 'R', 'G', 'B'])
```


## Training

To train the model call `SER_phonemes_learn.PhonemeSER.Train_model`:

```python
models_save_file = "data/model_file.pkl"
def Train_model(models_save_file, X_formants_train=features, Y_labels_train=labels[emotion], X_frame_lens_train=labels[length], inst_phoneme_types=[16,32,64], diff_phoneme_types=[16,32,64], K_SD=0, g_dist=8)
```
`Train_model` creates clustering models, selects phoneme features, trains classifiers and saves everything to a model file so that it can used later for testing.

### Parameters

`models_save_file`: string, file path to which clustering+classifying models are saved.

`X_formants_train`: array-like, shape = [n_clips, max_frames, n_formant_features]. Formant features for N utterences with fixed number of frames. Pass the actual frames length of each clip as array 'X_frame_lens_train' with the same order.

`Y_labels_train`: int array, shape = [n_clips]. Labels all clips in X_formants_train in the same order.

`X_frame_lens_train`: int array, shape = [n_clips]. Array of number of filled frames of each clip out of max_frames (max_frames default=800).

`inst_phoneme_types`: array-like, dtype=int16, shape = [n_models], optional (default=[16, 32, 64]). Cluster numbers for instantaneous (~25ms) phoneme clustering model. Set between 8 to 300 for each cluster model.

`diff_phoneme_types`:  array-like, dtype=int16, shape = [n_models], optional (default=[16, 32, 64]). Cluster numbers for differential phoneme (~25ms * g_dist) clustering models. Set between 8 to 300 for each cluster model.

`K_SD`: float, optional (default=0.0). Feature selection parameters. Set between -1 to 1. It sets the limit of standard deviation below the mean for selecting features within this threshold. Lower value selects more features.

`g_dist`: unsigned int, optional (default=8). Number for adjacent frames for measuring the change in formant features to calculate differential phoneme features.

### Returns

`Trained_classifiers`= `ClassifierObject()`, includes multiple trained classifiers, it doesn't need to be passed, because `Test_model` function reads this object from `models_save_file` created by `Train_model`.

## Testing

To test the model call `SER_phonemes_learn.PhonemeSER.Test_model`:

```python
def Test_model(models_save_file, X_formants_test, Y_labels_test, X_frame_lens_test)
```

Predict emotion classes using array-like features input of n_clips.

### Parameters

`models_save_file`: string, file path where clustering+classifying models are saved, use `Train_model()` to create a model file.

`X_formants_test`: array-like, shape = [n_clips, max_frames, n_formant_features]. Formant features for N utterences with fixed number of frames. Pass the actual frames length of each clip as array 'X_frame_lens_test' with the same order.

`Y_labels_test`: int array, shape = [n_clips]. Labels all clips in X_formants_test in the same order.

`X_frame_lens_test`: int array, shape = [n_clips]. Array of number of filled frames of each clip out of max_frames.

### Returns

`classifiers_results`: Dict list, shape: [{'classifier' : <classifier_name, string>, 'confusion' : <conf_matrix, array-like>, 'UAR' : <uar, float>, 'WAR' : <war, float>}].

`n_selected_features`: int, number of selected features.



## Validation

Run complete procedure with 'single-random' validation scheme:

```python
def run_train_and_test(db_names_paths, results_csv, val_scheme='single-random', test_size=0.2, folds_n=5, test_speakers=5, Nm_inst=[32, 64], Nm_diff=[32, 64], K_SD=0, g_dist=8, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, norm=0, deselect_labels=None, db_names_paths_2=None)
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

`emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies).

`norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters.

`deselect_labels`: list of chars, optional (default=None). Example: deselect_labels=['F', 'B'] to deselect 'F'and 'B'.

`db_names_paths_2`: Dict list of of test DBs, shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], optional (default=None). Same as 'db_names_paths', but only required for 'cross-corpus' validation scheme.

### Returns

`return 0` : on successful execution.

This function automatically assumes many parameters, e.g., classifiers (=all), formants HDF file name (autogenerated), model file name (autogenerated)


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
SVC1 H
SVC2 H
SVC3 H
SVC4 H
SVC5 A
RF02 A
RF01 H
RF03 H
RF04 H
KNN1 H
MLP1 A
Done
'''
```
------------------
