import numpy as np
import glob
import os

# DEMoS: an Italian emotional speech corpus. Elicitation methods, machine learning, and perception
# https://zenodo.org/record/2544829
'''
    EMOTIONS (three letters after the last '_'):
    Guilt - col
    Disgust - dis
    Happiness - gio
    Fear - pau
    Anger - rab
    Surprise - sor
    Sadness - tri
    Neutral - neu

'''

def create_DEMOS_file_objects(db_path):
  
    print("Creating file info objects")

    Sex_format = {"M":"m", "F":"f"}

    Emotions_format = {'N':'neu', 'H':'gio', 'S':'tri', 'A':'rab', 'F':'pau', 'D':'dis', 'U':'sor', 'G':'col'}



    array_of_clips = np.array([])

    for sex in Sex_format:

        for emotion in Emotions_format:

            for spkr_id in range(0, 70):
                speaker_id = str(spkr_id)
                if(spkr_id<10):
                    speaker_id = '0' + str(spkr_id)

                file_to_look = glob.glob(db_path +  'PR_' +  Sex_format[sex] + '_'+ speaker_id + '_' + Emotions_format[emotion]  + '*.wav')
                
                for index, wav_file_found in enumerate(file_to_look):
                    this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), 1, sex, emotion, intensity=1, statement=index, repetition=1, db_id=5)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    #print(wav_file_found)

        
    print("Total Clips", len(array_of_clips))
    return array_of_clips
    


