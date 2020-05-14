import SER_DB

'''
Speech Emotion Recognition (SER) databases augmentation

Currently supports these DBs
{'DB': "EmoDB", 'path' : "C:\\DB\\EMO-DB\\wav\\"},
{'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"},
{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},
{'DB': "ShemoDB", 'path' : "C:\\DB\\shemo\\"},
{'DB': "DEMoS", 'path' : "C:\\DB\\wav_DEMoS\\DEMOS\\"}
'''

# List of DB names and directory where wav files are stored:
db_names_paths = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},]
# Can add more than one DBs to the list




#Create a list of Clip_file_Class objects

list_of_augmented_clips = SER_DB.create_DB_file_objects(db_names_paths[0]['DB'], db_names_paths[0]['path'])

print("First file path:", list_of_augmented_clips[0].filepath)
print("First file emotion label:", list_of_augmented_clips[0].emotion)

