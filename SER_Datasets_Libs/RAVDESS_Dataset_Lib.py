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






def create_RAVDESS_file_objects(db_path):
    #For RAVDESS dataset
    print("Creating file info objects")

    Speaker_format = "Actor_*"


    #accent_format = {'1':'Speech', '2':'Song'}
    accent_format = {'1':'Speech'}
    Emotions_format = {'N':'01', 'C':'02', 'H':'03', 'S':'04', 'A':'05', 'F':'06', 'D':'07', 'U':'08'}
    intensity_format = {'1':'01', '2':'02'}
    statement_format = {'1':'01', '2':'02'}
    repetition_format = {'1':'01', '2':'02'}


    
    array_of_clips = np.array([])

    
    for accent in accent_format:

        for spkr_id in range(1, 25):
            speaker_id = str(spkr_id)
            if(spkr_id<10):
                speaker_id = '0' + str(spkr_id)
            sex = 'M'
            if((spkr_id % 2)==0):
                sex = 'F'

            for emotion in Emotions_format:
                for intensity in intensity_format:
                    for statement in statement_format:
                        for repetition in repetition_format:

                            file_to_look = glob.glob(db_path + accent_format[accent] + '\\Actor_' +  speaker_id + '\\03-*-' + 
                                Emotions_format[emotion] +  "-" +  intensity_format[intensity]  +  "-" + 
                                statement_format[statement]  +  "-" +  repetition_format[repetition]  + '-*.wav')
                            
                            for wav_file_found in file_to_look:
                                this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), int(accent),  sex, emotion, int(intensity), int(statement), int(repetition), db_id=1)
                                array_of_clips = np.append(array_of_clips, this_clip_info)
                                


    print("Total Clips", len(array_of_clips))
    return array_of_clips
    


