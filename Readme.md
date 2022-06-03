# Speech emotion recognition by phoneme type convergence



Predict speech emotions for small wav files.

Uses [formantfeatures](https://github.com/tabahi/formantfeatures) to get feature vectors for phoneme k-means clustering. Then Scikit-learn's classifiers are used to learn the emotional classification of based on what type of (learned) phonemes occur most in a clip.



## Installation

Version: 1.0.3

Install the python package from [pypip](https://pypi.org/project/PhonemeSER/)

```cmd
pip install PhonemeSER
```


## Pre-trained model

[Download pre-trained models from Onedrive](https://1drv.ms/u/s!Aht5RfGbNivUgU83aOBlJH1Nz7VC?e=gJCXqo) and run with `PhonemeSER.model_predict_wav_file(model_file, test_wav)`.

The biggest model `model_EmoD.RAVD.IEMO.Shem.DEMo.MSP` is 364 MB and performs at around 67% accuracy. Read more details about the performance in the paper [here](https://www.sciencedirect.com/science/article/abs/pii/S0020025521001584).


```python


import PhonemeSER


# Make sure pre-trained model file is downloaded and saved in the right path.
model_file = "../path/model_EmoD.RAVD.IEMO.Shem.DEMo.MSPI_single-random0.1_5_2_-0.5_16I32I64I12816D32D64D12860.0250.010.65_0.pkl"

test_wav = "263771femaleprotagonist.wav"

multi_classifiers_results = PhonemeSER.model_predict_wav_file(model_file, test_wav)
print(multi_classifiers_results)

'''
Output:
{'SVC1': 'S', 'SVC2': 'S', 'SVC3': 'S', 'SVC4': 'S', 'SVC5': 'S', 'RF01': 'S', 'RF03': 'N', 'RF04': 'S', 'KNN1': 'S', 'MLP1': 'S'}

where SVC1-MLP1 are different sklearn classifiers.
and S,N,H,A are labels for Sad, Neutral, Happy, and Angry respectively.
'''

```

## Pre-trained model performance


Testing log for model file: `model_EmoD.RAVD.IEMO.Shem.DEMo.MSPI_single-random0.1_5_2_-0.5_16I32I64I12816D32D64D12860.0250.010.65_0.pkl`. This model is trained on a combination of 6 multi-lingual datasets with 9:1 random split for training and validation.

```
data\Formants_EmoDB_25_10_650_0.hdf
data\Formants_RAVDESS_25_10_650_0.hdf
data\Formants_IEMOCAP_25_10_650_0.hdf
data\Formants_ShemoDB_25_10_650_0.hdf
data\Formants_DEMoS_25_10_650_0.hdf
data\Formants_MSPIMPROV_25_10_650_0.hdf
Reading dataset from file: ['data\\Formants_EmoDB_25_10_650_0.hdf', 'data\\Formants_RAVDESS_25_10_650_0.hdf', 'data\\Formants_IEMOCAP_25_10_650_0.hdf', 'data\\Formants_ShemoDB_25_10_650_0.hdf', 'data\\Formants_DEMoS_25_10_650_0.hdf', 'data\\Formants_MSPIMPROV_25_10_650_0.hdf']
Clips count: 11682
Total clips 11682
wav files size (MB) 3315.16
Total raw length (min) 763.85
Total trimmed length (min) 750.3
Avg raw length (s) 3.92
Avg trimmed length (s) 3.85
Avg. frame count 357.26
Male Female Clips 6285 5397
Unique speakers:  68
Speakers id:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
 49 50 51 52 53 54 55 56 57 58 59 60 61 63 64 65 66 67 68 69]
Emotion classes:  4
Unique emotions:  ['A', 'H', 'N', 'S']
Emotion N clips Total(min) Trimmed(min)
A        1846    113.63          110.23
H        3174    193.21          190.31
N        4743    317.52          312.05
S        1919    139.48          137.7
Validation scheme: single-random 0.1
Fold: 0 Train: 10513 Test: 1169
Training
Clustering (inst.) [16, 32, 64, 128] (10513, 800, 12)
Clustering (diff.) [16, 32, 64, 128] (10513, 800, 4)
Counting phonemes
Selecting features, K_SD: -0.5
Training Classifiers (samples, features): (10513, 481)
[65. 72. 78. 83.]
[1651. 2858. 4260. 1744.]
[4260. 4260. 4260. 4260.]
Testing
Using model file: data\model_EmoD.RAVD.IEMO.Shem.DEMo.MSPI_single-random0.1_5_2_-0.5_16I32I64I12816D32D64D12860.0250.010.65_0.pkl        Test samples: 1169
Classifiers: 10         Mean features: 481
SVC1    UAR: 69.82  WAR: 70.83  Tested samples: 1169
SVC2    UAR: 68.07  WAR: 66.38  Tested samples: 1169
SVC3    UAR: 60.19  WAR: 56.29  Tested samples: 1169
SVC4    UAR: 50.74  WAR: 45.59  Tested samples: 1169
SVC5    UAR: 67.7  WAR: 66.98   Tested samples: 1169
RF01    UAR: 63.77  WAR: 61.51  Tested samples: 1169
RF03    UAR: 62.1  WAR: 63.73   Tested samples: 1169
RF04    UAR: 67.94  WAR: 70.4   Tested samples: 1169
KNN1    UAR: 62.3  WAR: 59.28   Tested samples: 1169
MLP1    UAR: 66.71  WAR: 69.8   Tested samples: 1169
Finished 'run_train_and_test' for DB EmoD.RAVD.IEMO.Shem.DEMo.MSPI
```



------------


## SER model training

Training function `PhonemeSER.Train_model()` uses [K-means clustering (scikit-learn)](https://scikit-learn.org/stable/modules/clustering.html#k-means) to label the similar phonological units as the same phonemes. Then phoneme occurrences in whole DB are counted for each clip and occurrence rate is used to train [classifiers](https://scikit-learn.org/stable/modules/svm.html#svm-classification).

Once the model is trained and saved to a file, it can be tested by passing the testing set to `PhonemeSER.model_predict_wav_file(model_file, test_wav)`


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

[See SER_Datasets_Import module for reading database directory in their relevant filename formats.](https://github.com/tabahi/SER_Datasets_Import)
 Extract formant features and save to an HDF database by passing the DB name and path to `Extract_files_formant_features`:

```python
list_of_clips = SER_Datasets_Libs.SER_DB.create_DB_file_objects("DB_NAME", "C:/DB/path")

processed_n = FormantsLib.FormantsExtract.Extract_files_formant_features(list_of_clips, "formants_filename.hdf", window_length=0.025, window_step=0.01, f0_min=30, f0_max=4000, max_frames=800, formants=3)
```



Once formant features are extracted and saved to HDF they can be imported again to train the model:

```python
features, labels, u_speakers, u_classes  = HDFread.import_mutiple_HDFs(['formants_filename.hdf', 'file2.hdf', 'file3.hdf'], deselect_labels=['C', 'D', 'F', 'U', 'E', 'R', 'G', 'B'])
# multiple HDF files can be imported at once into the same numpy array - To import separately, call this function separately or call import_features_from_HDF
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

`diff_phoneme_types`:  array-like, dtype=int16, shape = [n_models], optional (default=[16, 32, 64]). Cluster numbers for differential phoneme (~10ms * g_dist) clustering models. Set between 8 to 300 for each cluster model.

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

`Nm_diff`: array-like, dtype=int16, shape = [n_models], optional (default=[32, 64]). Cluster numbers for differential phoneme (~10ms * g_dist) clustering models. Set between 8 and 300 for each cluster model.

`K_SD`: float, optional (default=0.0). Feature selection parameters. Set between -1 to 1. It sets the limit of standard deviation below the mean for selecting features within this threshold. Lower value selects more features.

`g_dist`: unsigned int, optional (default=8). Number for adjacent frames for measuring the change in formant features to calculate differential phoneme features.

`emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies).

`norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters.

`deselect_labels`: list of chars, optional (default=None). Example: deselect_labels=['F', 'B'] to deselect 'F'and 'B'.

`db_names_paths_2`: Dict list of of test DBs, shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], optional (default=None). Same as 'db_names_paths', but only required for 'cross-corpus' validation scheme.

### Returns

`return 0` : on successful execution.

This function automatically assumes many parameters, e.g., classifiers (=all), formants HDF file name (autogenerated), model file name (autogenerated)



------------------

## Citations

```tex
@article{LIU2021309,
title = {Speech emotion recognition based on formant characteristics feature extraction and phoneme type convergence},
journal = {Information Sciences},
volume = {563},
pages = {309-325},
year = {2021},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.02.016},
url = {https://www.sciencedirect.com/science/article/pii/S0020025521001584},
author = {Zhen-Tao Liu and Abdul Rehman and Min Wu and Wei-Hua Cao and Man Hao},
keywords = {Speech, Emotion recognition, Formants extraction, Phonemes, Clustering, Cross-corpus},
abstract = {Speech Emotion Recognition (SER) has numerous applications including human-robot interaction, online gaming, and health care assistance. While deep learning-based approaches achieve considerable precision, they often come with high computational and time costs. Indeed, feature learning strategies must search for important features in a large amount of speech data. In order to reduce these time and computational costs, we propose pre-processing step in which speech segments with similar formant characteristics are clustered together and labeled as the same phoneme. The phoneme occurrence rates in emotional utterances are then used as the input features for classifiers. Using six databases (EmoDB, RAVDESS, IEMOCAP, ShEMO, DEMoS and MSP-Improv) for evaluation, the level of accuracy is comparable to that of current state-of-the-art methods and the required training time was significantly reduced from hours to minutes.}
}
```

Paper: [Speech emotion recognition based on formant characteristics feature extraction and phoneme type convergence](https://www.sciencedirect.com/science/article/abs/pii/S0020025521001584)
