

import Train_Test

def main():

    #Few parameters:
    pt = [16, 32, 64, 128]  #total phoneme clusters, add more than one integer to this list to create multiple models
    K_SD = 0.0              #float, -1 to +1, feature selection parameters, less K_SD selects more featues

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
    

    Train_Test.run_train_and_test(db_names_paths, "results.csv", val_scheme='single-random', test_size=0.20, folds_n=5, Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=6, window_length=0.025, window_step=0.010, emphasize_ratio=0.65, deselect_labels=None)

        
    
    '''
    For cross-corpus validation
    '''
    db_names_paths = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},
                    {'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},] # Training sets
    db_names_paths_2 = [ {'DB': "EmoDB", 'path' : "C:\\DB\\EMO-DB\\wav\\"},] #Testing sets
    
    deselect_labels=['D','F','U','E','R', 'C', 'G', 'B']

    Train_Test.run_train_and_test(db_names_paths, "results.csv", val_scheme='cross-corpus', Nm_inst=pt, Nm_diff=pt, K_SD=K_SD, g_dist=8, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, deselect_labels=deselect_labels, db_names_paths_2=db_names_paths_2)
    

    '''
    For single file test
    '''
    test_wav = ".\\Archive\\test_F2.wav"   
    
    import SER_phonemes_learn.PhonemeSER as PhonemeSER
    # Make sure an already trained model file is available.
    model_file = "data/model_RAVD.IEMO_cross-corpus0.2_5_5_0.0_16I32I64I12816D32D64D12860.0250.010.65.pkl"
    PhonemeSER.Test_model_wav_file(model_file, test_wav)
    


    print("Done")
    exit()
    



if __name__ == '__main__':
    main()


