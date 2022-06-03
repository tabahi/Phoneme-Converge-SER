

import Train_Test

def main():

    #Few parameters:
    pt = [16, 32, 64, 128]  #total phoneme clusters, add more than one integer to this list to create multiple models
    K_SD = 0              #float, -1 to +1, feature selection parameters, less K_SD selects more featues
    norm = 0            #Normalization of mel-filter banks

    
    # List of DB names and directory where wav files are stored:
    db_names_paths = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},]
    # Can add more than one DBs to the list

    '''
    Currently supports these DBs
    {'DB': "EmoDB", 'path' : "C:\\DB\\EMO-DB\\wav\\"},
    {'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},
    {'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},
    {'DB': "ShemoDB", 'path' : "C:\\DB\\shemo\\"},
    {'DB': "DEMoS", 'path' : "C:\\DB\\wav_DEMoS\\DEMOS\\"}
    '''
    
    Train_Test.run_train_and_test(db_names_paths, "results.csv", val_scheme='k-folds', test_size=0.2, folds_n=5, Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=6, window_length=0.025, window_step=0.010, emphasize_ratio=0.65, norm=norm, deselect_labels=None)

    exit()

    
    # For cross-corpus validation:
    # Training sets:
    db_names_paths = [{'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},]

    #Testing sets
    db_names_paths_2 = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},]

    
    deselect_labels=['D','F','U','E','R', 'C', 'G', 'B']

    Train_Test.run_train_and_test(db_names_paths, "results.csv", val_scheme='cross-corpus', Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=6, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, norm=norm, deselect_labels=deselect_labels, db_names_paths_2=db_names_paths_2)
    

    


    print("Done")
    exit()
    



if __name__ == '__main__':
    main()


