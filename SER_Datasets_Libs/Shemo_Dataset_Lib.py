import numpy as np
import glob
import os


class Clip_file_Class(object):
    db_id = 1
    filepath = None
    speaker_id = None
    accent = None
    sex = None
    emotion = None
    intensity = None
    statement = None
    repetition = None

    def __init__(self, filepath, speaker_id, accent, sex, emotion, intensity, statement, repetition, db_id):
        self.filepath =  filepath
        self.speaker_id =  speaker_id
        self.accent =  accent
        self.sex =  sex
        self.emotion =  emotion
        self.intensity =  intensity
        self.statement =  statement
        self.repetition =  repetition
        self.db_id = db_id






'''
    M = male speaker
    F = female speaker
    S = sadness
    A = anger
    H = happiness
    W = surprise
    F = fear
    N = neutral

'''
#https://github.com/pariajm/Persian-Emotional-Speech-Database-ShEMO
def create_ShemoDB_file_objects(db_path):

    print("Creating file info objects")

    Sex_format = {"M":"M\\", "F":"F\\"}

    Emotions_format = {'N':'N', 'H':'H', 'S':'S', 'A':'A', 'F':'F', 'U':'W'}

    pre_format = '*'
    post_format = '*.wav'

    array_of_clips = np.array([])

    for sex in Sex_format:
        for spkr_id in range(0, 60):
            speaker_id = str(spkr_id)
            if(spkr_id<10):
                speaker_id = '0' + str(spkr_id)

            for emotion in Emotions_format:

                file_to_look = glob.glob(db_path +  Sex_format[sex]  + pre_format + speaker_id 
                    + Emotions_format[emotion]  + post_format)
                
                for index, wav_file_found in enumerate(file_to_look):
                    this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), 1,  sex, emotion, intensity=1, statement=index, repetition=1, db_id=3)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    
                    #break    
                #break
        
    print("Total Clips", len(array_of_clips))
    return array_of_clips
    


