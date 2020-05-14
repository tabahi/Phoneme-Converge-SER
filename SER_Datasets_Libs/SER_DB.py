
import numpy as np
import glob
import os
'''
 This module reads the wav files in DB folders to create clip file class objects
 All import functions read the DB folders in their default (downloaded as is from the source) format.
 Last updated by tabahi@hotmail.fr on 2020-05-05. Update: Organized all functions into one file.


 

The MIT License (MIT)

Copyright © 2020 Tabahi Abdul Rehman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

class Ix(object):
    #Clip label indices for enumeration - Ignore
    speaker_id, accent, sex, emotion, intensity, statement, repetition, frame_count, signal_len, trimmed_len, file_size =0,1,2,3,4,5,6,7,8,9,10
    

class Clip_file_Class(object):
    filepath = None
    speaker_id = None
    accent = None
    sex = None
    emotion = None
    intensity = None
    statement = None
    repetition = None
    db_id = 0

    frame_count = None
    signal_len = None
    trimmed_len = None
    file_size = None

    def __init__(self, filepath, speaker_id, accent, sex, emotion, intensity, statement, repetition, db_id, frame_count=None, signal_len=None, trimmed_len=None, file_size=None):
        self.filepath =  filepath
        self.speaker_id =  speaker_id
        self.accent =  accent
        self.sex =  sex
        self.emotion =  emotion
        self.intensity =  intensity
        self.statement =  statement
        self.repetition =  repetition
        self.db_id = db_id

        self.frame_count = frame_count
        self.signal_len = signal_len
        self.trimmed_len = trimmed_len
        self.file_size = file_size


def create_EMODB_file_objects(db_path):
    '''
    # http://emodb.bilderbar.info/docu/#docu
    '''

    print("Creating file info objects")

    Speakers_format = {'3':'03', '8':'08', '9':'09', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16'}
    Speakers_sex = {'3':'M', '8':'F', '9':'F', '10':'M', '11':'M', '12':'M', '13':'F', '14':'F', '15':'M', '16':'F'}
 
    Emotions_format = {'N':'N', 'H':'F', 'S':'T', 'A':'W', 'F':'A', 'D':'E', 'B': 'L'}

    statement_format = {'1':'a01', '2':'a02', '3':'a04', '4':'a05', '5':'a07', '6':'b01', '7':'b02', '8':'b03', '9':'b09', '10':'b10'}
    repetition_format = {'1':'a', '2':'b', '3':'c', '4':'d', '5':'e', '6':'f'}


    
    array_of_clips = np.array([])


    for speaker_id in Speakers_format:
        for emotion in Emotions_format:
            for statement in statement_format:
                for repetition in repetition_format:

                    file_to_look = glob.glob(db_path + Speakers_format[speaker_id] +
                        statement_format[statement] + Emotions_format[emotion] + repetition_format[repetition]  + '.wav')
                    
                    for wav_file_found in file_to_look:
                        this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), 1,  Speakers_sex[speaker_id], emotion, 1, int(statement), int(repetition), db_id=1)
                        array_of_clips = np.append(array_of_clips, this_clip_info)
                        #print(wav_file_found)


    print("Total Clips", len(array_of_clips))
    return array_of_clips




def create_RAVDESS_file_objects(db_path):
    '''
    For RAVDESS dataset
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
    '''
    print("Creating file info objects")
    #Speaker_format = "Actor_*"
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
                                this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), int(accent),  sex, emotion, int(intensity), int(statement), int(repetition), db_id=2)
                                array_of_clips = np.append(array_of_clips, this_clip_info)
                                


    print("Total Clips", len(array_of_clips))
    return array_of_clips



def create_IEMOCAP_file_objects(db_path):
    '''
    For IEMOCAP dataset
    https://sail.usc.edu/iemocap/iemocap_release.htm
    '''
    print("Creating file info objects")

    sessions = {'01':'Session1', '02':'Session2', '03':'Session3', '04':'Session4', '05':'Session5'}
    
    scenarios = {'1':'impro', '2':'script'}
    Emotions_format = {'N':'neu', 'H':'hap', 'S':'sad', 'A':'ang'} #'X':'xxx'
    # Excluded emotions: 'F':'fea', 'D':'dis', 'U':'sur', 'R':'fru', 'E':'exc'
    #Emotional content 10 cats - angry, happy, sad, neutral, frustrated, excited, fearful, disgusted, excited, other	

    array_of_clips = np.array([])

    for sess in sessions:
        for scene in scenarios:
            cat_files = glob.glob(db_path + sessions[sess] + '\\dialog\\EmoEvaluation\\Ses' + sess + '*_'+ scenarios[scene] + '*.txt')
            for cat_file in cat_files:
                #print(cat_file)
                with open(cat_file) as cat_file_txt:
                    content = cat_file_txt.readlines()
                # you may also want to remove whitespace characters like `\n` at the end of each line
                cat_file_lines = [x.strip() for x in content]
                for cat_line in cat_file_lines:
                    splitted = cat_line.split()
                    if(len(splitted) > 5):
                        for emo in Emotions_format:
                            if(splitted[4]==Emotions_format[emo]):
                                #print(splitted[3], splitted[4], splitted[3][-4], int(splitted[3][-3:]), splitted[3][0:-5])
                                speaker_id = ((int(sess)-1)*2) + (0 if (splitted[3][-4]=='F') else 1) + 1
                                #print(splitted[3][5], speaker_id)
                                
                                wav_file = db_path + sessions[sess] + '\\sentences\\wav\\' + splitted[3][0:-5] + '\\' + splitted[3] + '.wav'
                                #print(wav_file)
                                this_clip_info = Clip_file_Class(wav_file, speaker_id=speaker_id, accent=1, sex=splitted[3][-4], emotion=emo, intensity=1, statement=int(splitted[3][-3:]), repetition=1, db_id=3)
                                array_of_clips = np.append(array_of_clips, this_clip_info)
        

    print("Total Clips", len(array_of_clips))
    return array_of_clips





def create_ShemoDB_file_objects(db_path):
    '''
    M = male speaker
    F = female speaker
    S = sadness
    A = anger
    H = happiness
    W = surprise
    F = fear
    N = neutral
    
    #https://github.com/pariajm/Persian-Emotional-Speech-Database-ShEMO
    '''

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
                    this_clip_info = Clip_file_Class(wav_file_found, int(speaker_id), 1,  sex, emotion, intensity=1, statement=index, repetition=1, db_id=4)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    
                    #break    
                #break
        
    print("Total Clips", len(array_of_clips))
    return array_of_clips
    





def create_DEMOS_file_objects(db_path):
    
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
    # DEMoS: an Italian emotional speech corpus. Elicitation methods, machine learning, and perception
    # https://zenodo.org/record/2544829
    '''
  
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


def create_DB_file_objects(db_name, db_path):
    '''
    Creates a list of file objected of all the wav files present in the db_path, as long as they are according to the formant of relevant 'db_name's original source

    return list of Clip_file_Class objects

    Object Contains:
    (string)filepath, (int)speaker_id, (int)accent, (char)sex, (char)emotion, (int)intensity, (int)statement, (int)repetition, (int)db_id, (int)frame_count, (int)signal_len, (int)trimmed_len, (int)file_size
    
    Speech Emotion Recognition (SER) databases augmentation

    Currently supports these DBs

    db_name="EmoDB", db_path="C:\\DB\\EMO-DB\\wav\\";

    db_name="RAVDESS", db_path="C:\\DB\\RAVDESS\\";

    db_name="IEMOCAP", db_path="C:\\DB\\IEMOCAP_noVideo\\";

    db_name="ShemoDB", db_path="C:\\DB\\shemo\\";

    db_name="DEMoS", db_path="C:\\DB\\wav_DEMoS\\DEMOS\\";

    '''
    if(db_name=="EmoDB"):
        return create_EMODB_file_objects(db_path)

    elif(db_name=="RAVDESS"): 
        return create_RAVDESS_file_objects(db_path)
    elif(db_name=="IEMOCAP"): 
        return create_IEMOCAP_file_objects(db_path)
        
    elif(db_name=="ShemoDB"): 
        return create_ShemoDB_file_objects(db_path)
        
    elif(db_name=="DEMoS"): 
        return create_DEMOS_file_objects(db_path)
    
    else: raise Exception("Invalid DB name/key in db_sub_paths")



