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






def create_IEMOCAP_file_objects(db_path):

   '''
    For IEMOCAP dataset
    https://sail.usc.edu/iemocap/iemocap_release.htm
    '''
    print("Creating file info objects")

    sessions = {'01':'Session1', '02':'Session2', '03':'Session3', '04':'Session4', '05':'Session5'}
    
    scenarios = {'1':'impro', '2':'script'}
    Emotions_format = {'N':'neu', 'H':'hap', 'S':'sad', 'A':'ang', 'Y':'exc'} #'X':'xxx'
    # Add  'Y':'exc' for merging Happiness and Excitement
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
                            _emox = emo
                            if (emo == 'Y'):
                                _emox = 'H'
                            if(splitted[4]==Emotions_format[emo]):
                                #print(splitted[3], splitted[4], splitted[3][-4], int(splitted[3][-3:]), splitted[3][0:-5])
                                speaker_id = ((int(sess)-1)*2) + (0 if (splitted[3][-4]=='F') else 1) + 1
                                #print(splitted[3][5], speaker_id)
                                
                                wav_file = db_path + sessions[sess] + '\\sentences\\wav\\' + splitted[3][0:-5] + '\\' + splitted[3] + '.wav'
                                #print(wav_file)
                                this_clip_info = Clip_file_Class(wav_file, speaker_id=speaker_id, accent=1, sex=splitted[3][-4], emotion=_emox, intensity=1, statement=int(splitted[3][-3:]), repetition=1, db_id=3)
                                array_of_clips = np.append(array_of_clips, this_clip_info)
        

    print("Total Clips", len(array_of_clips))
    return array_of_clips


    


