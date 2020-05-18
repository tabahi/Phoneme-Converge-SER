import numpy as np
#import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.io
import os


import SER_Datasets_Libs.SER_DB as SER_DB
from FormantsLib.FormantsExtract import Extract_files_formant_features as Extract_formants
import FormantsLib.FormatsHDFread as HDFread
import SER_phonemes_learn.PhonemeSER as PhonemeSER

                              
def split_train_n_test(features, labels, scheme='single-random', test_size=0.2, folds_n=5, test_speakers=2, features_2=None, labels_2=None): 
    '''
    Parameters:

    features: array-like, dtype=int16, shape = [total_clips, max_frames, formants*4]. Array of clip frames formant features;

    labels: array-like, dtype=int16, shape = [total_clips, 11]. Each clip has 11 labels including speaker_id, emotions, accent, frame_count, trimmed_length etc.

    scheme : string. Choose one of these {'L1SO', 'LMSO',  'k-folds', 'single-random', 'cross-corpus'}, default: 'single-random',;

    test_speakers : only for scheme = 'LMSO', Leave Many Speakers Out;

    folds_n : only for scheme = 'k-folds';

    test_size : only for scheme = 'single-random';

    features_2, labels_2: Only for scheme = 'cross-corpus';

    returns: features_train, labels_train, features_test, labels_test

    '''

    features_train = []
    labels_train = []
    features_test = []
    labels_test = []
    if(scheme=='L1SO'):
        
        print("Validation scheme:", 'L1SO')
        unique_speakers = np.unique(labels[:, SER_DB.Ix.speaker_id])

        for s_id in range(len(unique_speakers)):
            select_train = np.where(labels[:, SER_DB.Ix.speaker_id]!=unique_speakers[s_id])
            select_test = np.where(labels[:, SER_DB.Ix.speaker_id]==unique_speakers[s_id])

            features_train.append(features[select_train] )
            labels_train.append(labels[select_train] )
            features_test.append(features[select_test] )
            labels_test.append(labels[select_test])
            #print ("Test Speaker:", unique_speakers[s_id], "Train:", features_train[s_id].shape[0], "Test:", features_test[s_id].shape[0])
    elif(scheme=='LMSO'):
        
        print("Validation scheme:", 'LMSO', test_speakers)
        unique_speakers = np.unique(labels[:, SER_DB.Ix.speaker_id])
        total_speakers = len(unique_speakers)
        folds = int(round(total_speakers/test_speakers))

        for s_id in range(folds):
            stx = test_speakers*s_id
            edx = stx + test_speakers
            if edx > total_speakers: edx = total_speakers
            this_fold_speakers = unique_speakers[stx:edx]

            conditions = labels[:, SER_DB.Ix.speaker_id] == this_fold_speakers[0]
            for ts in this_fold_speakers:
                conditions1 =  (labels[:, SER_DB.Ix.speaker_id] == ts)
                conditions = conditions | conditions1
            
            select_train = np.where(conditions != True)
            select_test = np.where(conditions == True)

            features_train.append(features[select_train] )
            labels_train.append(labels[select_train] )
            features_test.append(features[select_test] )
            labels_test.append(labels[select_test])
            print ("Test speakers:", this_fold_speakers, "Train:", features_train[s_id].shape[0], "Test:", features_test[s_id].shape[0])
            
    elif(scheme=='k-folds'):
        print("Validation scheme:", 'k-folds', folds_n)
        from sklearn.model_selection import ShuffleSplit
        rs = ShuffleSplit(n_splits=folds_n, test_size=1/folds_n, random_state=0)
        for select_train, select_test in rs.split(features):
            
            features_train.append(features[select_train] )
            labels_train.append(labels[select_train] )
            features_test.append(features[select_test] )
            labels_test.append(labels[select_test])
            #print ("Fold:", "Train:", features[select_train].shape[0], "Test:", features[select_test].shape[0])

    elif(scheme=='single-random'):
        print("Validation scheme:", 'single-random', test_size)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size = test_size, random_state = 100)
    
        features_train.append(X_train)
        labels_train.append(Y_train)
        features_test.append(X_test)
        labels_test.append(Y_test)

    elif(scheme=='cross-corpus'):
        print("Validation scheme:", 'cross-corpus')
        if (features_2.all()!=None and labels_2.all()!=None):
            print("Whole DB_1 is (X_train, Y_train), DB_2 is (X_test, Y_test)")

            features_train.append(features)
            labels_train.append(labels)
            features_test.append(features_2)
            labels_test.append(labels_2)
        else:
            raise Exception ("Cross-corpus missing arguments: 'features_2' and 'labels_2'")


    return features_train, labels_train, features_test, labels_test


def save_results_csv(results_csv, db_names, classifier_name, uar, war, scheme, test_size, folds_n, K_SD, features_count, Nm_inst, Nm_diff, g_dist, window_length, window_step, emphasize_ratio, classes_n, speakers_n, samples, confusion):
    import csv

    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    if(os.path.isfile("results\\" + results_csv)==False):
        with open("results\\" + results_csv, 'w') as csvFile:
            csvFile.write("db_names, classifier, UAR, WAR, scheme, test_size, folds_n, K_SD, features_count, Nm_inst, Nm_diff, g_dist, window_length, window_step, emphasize_ratio, classes_n, speakers_n, samples, confusion\n")

    with open("results\\" + results_csv, 'a') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', lineterminator = '\n')
        #writer.writerow(["Emotion", "Combination", "Occurrences"])
        this_row = [db_names, classifier_name, str(uar), str(war), scheme, str(test_size), str(folds_n), str(K_SD), 
                str(features_count), str(Nm_inst), str(Nm_diff),
                str(g_dist), str(window_length*1000), str(window_step*1000), str(emphasize_ratio), 
                str(classes_n), str(speakers_n), str(samples), str(confusion)]


        writer.writerow(this_row)

def make_formants_filename(db_name, window_length, window_step, emphasize_ratio, norm):
    if not os.path.exists('data'):
        os.makedirs('data')
    return 'data\\Formants_' + db_name + '_'  + str(int(window_length*1000)) + '_' + str(int(window_step*1000)) + '_' + str(int(emphasize_ratio*1000)) + '_' + str(norm) + '.hdf'

def make_model_filename(DB_names, val_scheme, test_size, folds_n, test_speakers, K_SD, Nm_inst, Nm_diff, g_dist, window_length, window_step, emphasize_ratio, norm):
    if not os.path.exists('data'):
        os.makedirs('data')
    return "data\\model_" + DB_names + "_" + str(val_scheme) + str(test_size)+ "_"+ str(folds_n) + "_"+ str(test_speakers) +"_" + str(K_SD) + "_" + "I".join(str(x) for x in Nm_inst) + "D".join(str(x) for x in Nm_diff) + str(g_dist) + str(window_length) + str(window_step) + str(emphasize_ratio)+ '_' + str(norm)+ ".pkl"



def run_train_and_test(db_names_paths, results_csv, val_scheme='single-random', test_size=0.2, folds_n=5, test_speakers=5, Nm_inst=[32, 64], Nm_diff=[32, 64], K_SD=0, g_dist=8, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, norm=0, deselect_labels=None, db_names_paths_2=None):
    '''
    Run training and testing functions with the set parameters

    Parameters
    ----------

    `db_names_paths`: Dict list of names and paths of DBs. Shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], Choose db_name from these: {"EmoDB", "RAVDESS", "IEMOCAP", "ShemoDB", "DEMoS"}
    with respective directory paths of the DB wav files. Importing of each DB is programed in 'Dataset_Lib/SER_DB.py'. The organization format of annotated wav files is expected to be as is downloaded from their original sources. For val_scheme='cross-corpus', all these DBs are used as training sets and `db_names_paths_2` must be passed as testing DB list.

    `results_csv`: string, optional (default=None), CSV file path to which final results will be appended.

    `val_scheme`: {'L1SO', 'LMSO', 'single-random', 'k-folds', 'cross-corpus'}, optional (default=single-random). Select one of these validation schemes: 'L1SO' : Leave One Speaker Out, 'LMSO': Leave multiple speakers out (default: 5 speakers, require parameter 'test_speakers'), 'single-random': Single random split (requires param 'test_size'), 'k-folds': K-folds cross-validation (require param: 'folds_n').

    `test_size`: float, optional (default=0.2). Fraction of total clips to use as testing set for 'single-random' validation scheme. Ignored for other schemes.

    `folds_n`: unsigned int, optional (default=5). Number of folds for 'k-folds' validation scheme. Ignored for other schemes.

    `test_speakers`: unsigned int, optional (default=5). Number of test speakers for 'LMSO' validation scheme. Ignored for other schemes.

    `Nm_inst`: array-like, dtype=int16, shape = [n_models], optional (default=[32, 64]). Cluster numbers for instantaneous (~25ms) phoneme clustering model. Set between 8 to 300 for each cluster model.
    
    `Nm_diff`:  array-like, dtype=int16, shape = [n_models], optional (default=[32, 64]). Cluster numbers for differential phoneme (~25ms * g_dist) clustering models. Set between 8 to 300 for each cluster model.

    `K_SD`: float, optional (default=0.0). Feature selection parameters. Set between -1 to 1. It sets the limit of standard deviation below the mean for selecting features within this threshold. Lower value selects more features.

    `g_dist`: unsigned int, optional (default=8). Number for adjacent frames for measuring the change in formant features to calculate differential phoneme features.
    
    `emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies).

    `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters.

    `deselect_labels`: list of chars, optional (default=None). Example: deselect_labels=['F', 'B'] to deselect 'F'and 'B'.

    `db_names_paths_2`: Dict list of of test DBs, shape: [{'DB':'<db_name>', 'path': '<dir_path>'},], optional (default=None). Same as 'db_names_paths', but only required for 'cross-corpus' validation scheme.

    Returns
    -------

    `return 0` : on succussful execution.
    '''
    
    #list of HDF storage file paths in which formant characteristics are stored
    features_HDF_files = []
    features_test_HDF_files = []

    
    DB_names = ".".join([db_names_paths[x]['DB'][0:4] for x in range(0, len(db_names_paths))])

    

    # Create the model filename using all the parameters.
    models_save_file = make_model_filename(DB_names, val_scheme, test_size, folds_n, test_speakers, K_SD, Nm_inst, Nm_diff, g_dist, window_length, window_step, emphasize_ratio, norm)

    
    
    #check if features are already extracted
    for db_name_path in db_names_paths:
        #HDF storage file path in which formant characteristics are stored
        features_HDF_file = make_formants_filename(db_name_path['DB'], window_length, window_step, emphasize_ratio, norm)

        if (os.path.isfile(features_HDF_file)==False) or (int(os.path.getsize(features_HDF_file))<8000):
            array_of_clips = SER_DB.create_DB_file_objects(db_name_path['DB'], db_name_path['path'])
            #Extract and save formant features of clips in array to an HDF file along with labels (labels are included in file_objects)
            processed_n = Extract_formants(array_of_clips, features_HDF_file, window_length, window_step, emphasize_ratio, norm, f0_min=30, f0_max=4000, max_frames=800, formants=3)
            if(processed_n==0):
                raise Exception("No files to process. Make sure DB directory path and filenames are in the correct format.")
        
        features_HDF_files.append(features_HDF_file)

    #Read formant features and labels from the HDF file
    features, labels, u_speakers, u_classes  = HDFread.import_mutiple_HDFs(features_HDF_files, deselect_labels=deselect_labels)


    HDFread.print_database_stats(labels)

    features2, labels2 = None, None
    #Only for cross-corpus validation scheme
    if(val_scheme=='cross-corpus'):
        if(len(db_names_paths_2)>0):
            for db_name_path in db_names_paths_2:
                features_HDF_file = make_formants_filename(db_name_path['DB'], window_length, window_step, emphasize_ratio, norm)

                if (os.path.isfile(features_HDF_file)==False) or (int(os.path.getsize(features_HDF_file))<8000):
                    array_of_clips = SER_DB.create_DB_file_objects(db_name_path['DB'], db_name_path['path'])
                    processed_n = Extract_formants(array_of_clips, features_HDF_file, window_length, window_step, emphasize_ratio, f0_min=30, f0_max=4000, max_frames=800, formants=3)
                    if(processed_n==0):
                        raise Exception("No files to process. Make sure DB directory path and filenames are in the correct format.")
                features_test_HDF_files.append(features_HDF_file)
            features2, labels2, u_speakers2, u_classes2  = HDFread.import_mutiple_HDFs(features_test_HDF_files, deselect_labels=deselect_labels)
            print("Train set labels:", [chr(x) for x in u_classes])
            #HDFread.print_database_stats(labels1)
            print("Test set labels:", [chr(x) for x in u_classes])
            DB_names += "_"
            DB_names += ".".join([db_names_paths_2[x]['DB'][0:4] for x in range(0, len(db_names_paths_2))])
        else:
            raise Exception ("Missing argument 'db_names_paths_2'")




    folds_results = []
    features_counts = []
    
    Xtr, Ytr, Xts, Yts = split_train_n_test(features, labels, scheme=val_scheme, test_speakers=5, folds_n=folds_n, test_size=test_size, features_2=features2, labels_2=labels2)

    for fold in range(0, len(Xtr)):
        print ("Fold:", fold, "Train:", Xtr[fold].shape[0], "Test:", Xts[fold].shape[0])

        print("Training")
        if((val_scheme!='cross-corpus') or ((val_scheme=='cross-corpus') and (os.path.isfile(models_save_file)==False))):
            PhonemeSER.Train_model(models_save_file, Xtr[fold], Ytr[fold][:, SER_DB.Ix.emotion], Ytr[fold][:, SER_DB.Ix.frame_count], Nm_inst, Nm_diff, K_SD, g_dist)
        else:
            print("Skipping training because model file already exists.")

        print("Testing")
        print("Using model file:", models_save_file, "\t Test samples:", Xts[fold].shape[0])
        classifiers_results, ft_count = PhonemeSER.Test_model(models_save_file, Xts[fold], Yts[fold][:, SER_DB.Ix.emotion], Yts[fold][:, SER_DB.Ix.frame_count])
        
    
        folds_results.append(classifiers_results)
        features_counts.append(ft_count)
    
    features_count = int(np.mean(features_counts)) #mean of features counts of each fold

    #classifiers =  [folds_results[x]['classifier'] for x in range(0, len(folds_results[0]))]
    print("Folds:", len(folds_results), "\rClassifiers:", len(folds_results[0]), "\tMean features:", features_count)
    
    for c in range(len(folds_results[0])):
        sum_conf = np.zeros_like(folds_results[0][c]['confusion'])
        for fold in range(len(folds_results)):
            sum_conf += folds_results[fold][c]['confusion']
    
        
            
        samples = np.sum(sum_conf)
        print(folds_results[0][c]['classifier'], "\tUAR:", folds_results[0][c]['UAR'], " WAR:", folds_results[0][c]['WAR'], "\tTested samples:", samples)
        
        if(results_csv!=None):
            save_results_csv(results_csv, DB_names, folds_results[0][c]['classifier'], folds_results[0][c]['UAR'], folds_results[0][c]['WAR'], val_scheme, test_size, folds_n, K_SD, features_count, Nm_inst, Nm_diff, g_dist, window_length, window_step, emphasize_ratio, len(u_classes), len(u_speakers), samples, sum_conf)

    print("Finished 'run_train_and_test' for DB", DB_names)

    return 0





def main():

    #Few parameters:
    pt = [16, 32, 64, 128]  #total phoneme clusters, add more than one integer to this list to create multiple models
    K_SD = 0              #float, -1 to +1, feature selection parameters, less K_SD selects more featues
    norm = 0            #Normalization of mel-filter banks

    
    # List of DB names and directory where wav files are stored:
    db_names_paths = [{'DB': "EmoDB", 'path' : "C:\\DB\\EMO-DB\\wav\\"},]
    # Can add more than one DBs to the list

    '''
    Currently supports these DBs
    {'DB': "EmoDB", 'path' : "C:\\DB\\EMO-DB\\wav\\"},
    {'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},
    {'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},
    {'DB': "ShemoDB", 'path' : "C:\\DB\\shemo\\"},
    {'DB': "DEMoS", 'path' : "C:\\DB\\wav_DEMoS\\DEMOS\\"}
    '''
    
    run_train_and_test(db_names_paths, "results.csv", val_scheme='k-folds', test_size=0.2, folds_n=5, Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=6, window_length=0.025, window_step=0.010, emphasize_ratio=0.65, norm=norm, deselect_labels=None)

    exit()

    
    # For cross-corpus validation:
    # Training sets:
    db_names_paths = [{'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},]

    #Testing sets
    db_names_paths_2 = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},]

    
    deselect_labels=['D','F','U','E','R', 'C', 'G', 'B']

    run_train_and_test(db_names_paths, "results.csv", val_scheme='cross-corpus', Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=6, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, norm=norm, deselect_labels=deselect_labels, db_names_paths_2=db_names_paths_2)
    

    


    print("Done")
    exit()
    



if __name__ == '__main__':
    main()

