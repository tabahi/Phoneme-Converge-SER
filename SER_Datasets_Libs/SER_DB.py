'''
 This module reads the wav files in DB folders to create clip file class objects
 All import functions read the DB folders in their default (downloaded as is from the source) format.
 Last updated by tabahi@hotmail.fr on 2020-05-05. Update: Organized all functions into one file.

sex: 'M', 'F'

scenario: 0=unknown, 1=script, 2=improv, 3=radio/TV, 4=elicited, 5=natural, 6=script-in-improv

emotion_cat: {'N':'neutral', 'H':'happy', 'S':'sad', 'A':'anger',  'F':'fear', 'D':'disgust', 'U':'surprise', 'C':'calm', 'R':'frustuated', 'E':'excited', 'Y':'happy-excited', 'G':'guilty', 'X': 'unknown'}


2020-05-24 Tabahi Abdul Rehman


'''

import numpy as np
import glob
import os


class Clip_file_Class(object):

    db_id = 0            # integer
    filepath = None      # string
    speaker_id = None    # integer
    scenario = None      # integer

    sex = None          # char
    emotion_cat = None      # categorical
    intensity_cat = None    # categorical

    valance = None      # dimensional
    arousal = None      # dimensional
    dominance = None    # dimensional
    naturalness = None  # dimensional

    statement = None    # integer
    repetition = None   # integer

    frame_count = None       # integer
    signal_len = None        # integer
    trimmed_len = None       # integer
    file_size = None         # integer
    
    n_raters = None
    n_possible_emotions = None
    raters_labels_cat = None # includes emotion_cat (0 or 1 for each category) by each rater
    raters_labels_dim = None # includes valance, arousal, dominance by each rater

    def __init__(self, db_id, filepath, speaker_id, scenario, sex, emotion_cat=None, intensity_cat=None, valance=None, arousal=None, dominance=None, naturalness=None, statement=None, repetition=None, n_raters=None, n_possible_emotions=None):
        # Initialize with ground truth lables
        self.db_id = db_id
        self.filepath =  filepath
        self.speaker_id =  speaker_id
        self.scenario =  scenario
        self.sex =  sex

        self.emotion_cat =  emotion_cat
        self.intensity_cat =  intensity_cat

        self.valance = valance
        self.arousal = arousal
        self.dominance = dominance
        self.naturalness = naturalness

        self.statement =  statement
        self.repetition =  repetition
        
        if(n_raters):
            self.n_raters = n_raters
            self.n_possible_emotions = n_possible_emotions
            self.raters_labels_cat = np.zeros((n_raters, self.n_possible_emotions,), dtype=np.uint8)
            self.raters_labels_dim = np.zeros((n_raters, 4,), dtype=np.uint8)




class Ix(object):
    #Clip label indices for enumeration - Ignore
    speaker_id, scenario, sex, emotion_cat, intensity_cat, statement, repetition, frame_count, signal_len, trimmed_len, file_size =0,1,2,3,4,5,6,7,8,9,10
    

def create_EMODB_file_objects(db_path, deselect=None):
    '''
    # http://emodb.bilderbar.info/docu/#docu
    '''

    print("Creating file info objects")

    Speakers_format = {'3':'03', '8':'08', '9':'09', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16'}
    Speakers_sex = {'3':'M', '8':'F', '9':'F', '10':'M', '11':'M', '12':'M', '13':'F', '14':'F', '15':'M', '16':'F'}
 
    emotion_cats_format = {'N':'N', 'H':'F', 'S':'T', 'A':'W', 'F':'A', 'D':'E', 'B': 'L'}

    statement_format = {'1':'a01', '2':'a02', '3':'a04', '4':'a05', '5':'a07', '6':'b01', '7':'b02', '8':'b03', '9':'b09', '10':'b10'}
    repetition_format = {'1':'a', '2':'b', '3':'c', '4':'d', '5':'e', '6':'f'}


    
    array_of_clips = np.array([])


    for speaker_id in Speakers_format:
        for emotion_cat in emotion_cats_format:
            for statement in statement_format:
                for repetition in repetition_format:

                    file_to_look = glob.glob(db_path + Speakers_format[speaker_id] +
                        statement_format[statement] + emotion_cats_format[emotion_cat] + repetition_format[repetition]  + '.wav')
                    
                    for wav_file_found in file_to_look:
                        this_clip_info = Clip_file_Class(1, wav_file_found, int(speaker_id), 1,  Speakers_sex[speaker_id], emotion_cat, 1, None,None, None, None, int(statement), int(repetition))
                        array_of_clips = np.append(array_of_clips, this_clip_info)
                        #print(wav_file_found)


    print("Total Clips", len(array_of_clips))
    if(len(array_of_clips)<1): raise Exception("No clips found")  
    return array_of_clips




def create_RAVDESS_file_objects(db_path, deselect=None):
    '''
    For RAVDESS dataset
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
    '''
    print("Creating file info objects")
    #Speaker_format = "Actor_*"
    #scenario_format = {'1':'Speech', '2':'Song'}
    scenario_format = {'1':'Speech'}
    emotion_cats_format = {'N':'01', 'C':'02', 'H':'03', 'S':'04', 'A':'05', 'F':'06', 'D':'07', 'U':'08'}
    intensity_cat_format = {'1':'01', '2':'02'}
    statement_format = {'1':'01', '2':'02'}
    repetition_format = {'1':'01', '2':'02'}

    array_of_clips = np.array([])
    
    for scenario in scenario_format:

        for spkr_id in range(1, 25):
            speaker_id = str(spkr_id)
            if(spkr_id<10):
                speaker_id = '0' + str(spkr_id)
            sex = 'M'
            if((spkr_id % 2)==0):
                sex = 'F'

            for emotion_cat in emotion_cats_format:
                for intensity_cat in intensity_cat_format:
                    for statement in statement_format:
                        for repetition in repetition_format:

                            file_to_look = glob.glob(db_path + scenario_format[scenario] + '\\Actor_' +  speaker_id + '\\03-*-' + 
                                emotion_cats_format[emotion_cat] +  "-" +  intensity_cat_format[intensity_cat]  +  "-" + 
                                statement_format[statement]  +  "-" +  repetition_format[repetition]  + '-*.wav')
                            
                            for wav_file_found in file_to_look:
                                #print(wav_file_found, emotion_cat)
                                this_clip_info = Clip_file_Class(2, wav_file_found, int(speaker_id), int(scenario),  sex, emotion_cat, int(intensity_cat), None,None, None, None, int(statement), int(repetition))
                                array_of_clips = np.append(array_of_clips, this_clip_info)
                                


    print("Total Clips", len(array_of_clips))
    if(len(array_of_clips)<1): raise Exception("No clips found")

    #print(array_of_clips[1].filepath, array_of_clips[1].emotion_cat)
    
    
    return array_of_clips



def create_IEMOCAP_file_objects(db_path, deselect=['F', 'D', 'U', 'R', 'X']):
    '''
    For importing IEMOCAP dataset (wav files only)
    https://sail.usc.edu/iemocap/iemocap_release.htm

    + Evaluation files path formant:  '<db_path>\\Session<1>\\dialog\\EmoEvaluation\\Ses<01><F>_ <SCENARIOS>*.txt'

    + WAV files path format: '<db_path>\\Session<1>\\sentences\\wav\\ <PATH_FROM_EVALUATION_FILE> .wav'

    + Merges Happiness and Excitement
    '''
    print("Creating file info objects")
    
    sessions = {'01':'Session1', '02':'Session2', '03':'Session3', '04':'Session4', '05':'Session5'}
    
    scenarios = {'1':'script','2':'impro'}
    emotion_cats_format = {'N':'neu', 'H':'hap', 'S':'sad', 'A':'ang',  'F':'fea', 'D':'dis', 'U':'sur', 'R':'fru', 'Y':'exc', 'X':'xxx'}
    # Add  'Y':'exc' for merging Happiness and Excitement
    # Excluded emotion_cats: 'F':'fea', 'D':'dis', 'U':'sur', 'R':'fru', 'E':'exc'
    #emotion_catal content 10 cats - angry, happy, sad, neutral, frustrated, excited, fearful, disgusted, excited, other	

    raters_id = {'C-E1':0, 'C-E2': 1, 'C-E3': 2, 'C-E4': 3, 'C-E5':4,'C-E6':5, 'C-F1': 6, 'C-F2': 7, 'C-F3': 8, 'C-M1': 9, 'C-M3': 10, 'C-M5': 11,  'A-E1': 12, 'A-E2': 13,'A-E3': 14, 'A-E4': 15, 'A-E5': 16, 'A-E6': 17, 'A-F1': 18,  'A-F2': 19, 'A-F3': 20, 'A-M1': 21, 'A-M3': 22, 'A-M5': 23}

    
    array_of_clips = np.array([])

    for sess in sessions:
        for scene in scenarios:

            cat_files = glob.glob(db_path + sessions[sess] + '\\dialog\\EmoEvaluation\\Ses' + sess + '*_'+ scenarios[scene] + '*.txt')
            for cat_file in cat_files:
                #print(cat_file)
                with open(cat_file) as cat_file_txt:
                    content = cat_file_txt.readlines()
                
                cat_file_lines = [x.strip() for x in content]

                for line_no in range(1, len(cat_file_lines)):
                    cat_line = cat_file_lines[line_no]
                    splitted = cat_line.split()

                    if (len(cat_file_lines[line_no-1]) < 2) &  (len(splitted) > 5): # detect first line of each entry
                        for emo in emotion_cats_format:
                            

                            if(splitted[4]==emotion_cats_format[emo]) and  (emo not in deselect):
                                _emox = emo
                                if (emo == 'Y'):    #change excitement to happy
                                    _emox = 'H'
                                    
                                emotion_cat = _emox
                                valance = float(''.join(i for i in splitted[-3] if i.isdigit() or (i=='.') ))
                                arousal = float(''.join(i for i in splitted[-2] if i.isdigit() or (i=='.') ))
                                dominance = float(''.join(i for i in splitted[-1] if i.isdigit() or (i=='.') ))



                                # Each session has 2 speakers, we assign: Female = SessionID+0, Male = SessionID+1, 
                                speaker_id = ((int(sess)-1)*2) + (0 if (splitted[3][-4]=='F') else 1) + 1
                                
                                
                                wav_file = os.path.join(db_path, sessions[sess] + '\\sentences\\wav\\' + splitted[3][0:-5] + '\\' + splitted[3] + '.wav')
                                
                                this_clip_info = Clip_file_Class(3, wav_file, speaker_id=speaker_id, scenario=int(scene), sex=splitted[3][-4], emotion_cat=emotion_cat, intensity_cat=1, valance=valance, arousal=arousal, dominance=dominance, statement=int(splitted[3][-3:]), repetition=1, n_raters=len(raters_id), n_possible_emotions=len(emotion_cats_format))
                                
                                # parse individual ratings
                                next_line = line_no + 1
                                while(next_line < len(cat_file_lines) - 1) and (len(cat_file_lines[next_line])>10) and  (cat_file_lines[next_line][0] in ['C' , 'A']):
                                    splits_2 = cat_file_lines[next_line].split()
                                    
                                    rater_id = raters_id[splits_2[0][0:-1]]

                                    if(splits_2[0][0]=='C'):
                                        
                                        for e_index, emo in enumerate(emotion_cats_format):
                                            
                                            for this_split in splits_2:
                                                if (emotion_cats_format[emo] in this_split.lower()):
                                                    this_clip_info.raters_labels_cat[rater_id, e_index] = 1
         
                                    elif(splits_2[0][0]=='A'):

                                        r_valance = int(''.join(i for i in splits_2[2] if i.isdigit() ))
                                        r_arousal = int(''.join(i for i in splits_2[4] if i.isdigit() ))
                                        r_dominance = 0
                                        if(len(splits_2[6])>1):
                                            r_dominance = int('0'.join(i for i in splits_2[6] if i.isdigit() ))
                                        

                                        this_clip_info.raters_labels_dim[rater_id] = np.array([r_valance, r_arousal, r_dominance, 0], np.uint8)
                                        
                                    
                                    next_line += 1

                                #rate_mean = np.mean(this_clip_info.raters_labels_cat, axis=0)
                                #rate_mean = rate_mean / np.sum(rate_mean)
                                
                                #if( len(np.where(rate_mean>=0.667)[0])==1 ) : 
                                array_of_clips = np.append(array_of_clips, this_clip_info)

                                break

    print("Total Clips", len(array_of_clips))
    
    if(len(array_of_clips)<1): raise Exception("No clips found")  
    return array_of_clips


def create_MSPIMPROV_file_objects(db_path, deselect=None):

    print("Creating file info objects")

    array_of_clips = np.array([])

    #scenario: 0=unknown, 1=script, 2=improv, 3=radio/TV, 4=elicited, 5=natural, 6=script-in-improv
    scenarios = {'R':1, 'S':2, 'P':5, 'T':6}

    emotion_cats_format = {'N':'N', 'H':'H', 'S':'S', 'A':'A',} #main emotions that are already complied from averaging of individual raters
    raters_cats_format = {'N':'Neutral', 'H':'Happy', 'S':'Sad', 'A':'Angry', 'U': 'Surprised', 'E' : 'Excited', 'F' : 'Fearful', 'P' : 'Depressed', 'R' :'Frustrated', 'D' : 'Disgusted', 'X' : 'Other'}   #these are used to compile the rating my own way


    eval_file = os.path.join(db_path, "Evalution.txt")
        
    with open(eval_file) as eval_file_txt:
        content = eval_file_txt.readlines()
    
    eval_file_lines = [x.strip() for x in content]

    for line_no in range(0, len(eval_file_lines)):
        
        if(".avi;" in eval_file_lines[line_no]):
            
            splitted = eval_file_lines[line_no].split(';')

            file_labels = splitted[0][3:-4].split('-')
            sex = file_labels[3][0:1]
            session_no = int(file_labels[3][-2:])
            sentence = int(file_labels[2][1:-1])
            scene = file_labels[4][0:1]
            emotion = splitted[1].strip()

            turn = int(file_labels[-1][-2:])
            

            valance = float(''.join(i for i in splitted[2] if i.isdigit() or (i=='.') ))
            arousal = float(''.join(i for i in splitted[3] if i.isdigit() or (i=='.') ))
            dominance = None
            try:
                dominance = float(''.join(i for i in splitted[4] if i.isdigit() or (i=='.') ))
            except:
                pass
            naturalness = None
            try:
                naturalness = float(''.join(i for i in splitted[5] if i.isdigit() or (i=='.') ))
            except:
                pass
                
            # Each session has 2 speakers, we assign: Female = SessionID+0, Male = SessionID+1, 
            speaker_id = ((int(session_no)-1)*2) + (0 if (sex=='F') else 1) + 1
            
            
            #print(session_no, scenario, sex, emotion, valance, arousal, dominance, naturalness)
            wav_files = glob.glob(db_path + 'session' + str(session_no) + '\\S*'+ emotion +'\\' + scene +'\\MSP' +splitted[0][3:-4]+ '.wav')

            for wav_file in wav_files:
                this_clip_info = Clip_file_Class(3, wav_file, speaker_id=speaker_id, scenario=int(scenarios[scene]), sex=sex, emotion_cat=emotion_cats_format[emotion], intensity_cat=1, valance=valance, arousal=arousal, dominance=dominance, naturalness=naturalness, statement=sentence, repetition=turn)
            
            rate_mean = np.zeros((len(raters_cats_format),))
            # parse individual ratings
            next_line = line_no + 1
            while(next_line < len(eval_file_lines) - 1) and (len(eval_file_lines[next_line])>10) and  ("-p" in eval_file_lines[next_line]):
                splits_2 = eval_file_lines[next_line].split(';')
                
                for index, emo_types in enumerate(raters_cats_format):
                    if(raters_cats_format[emo_types] in splits_2[1].strip()):
                        rate_mean[index] += 1
                #exit()
                next_line += 1
                
            rate_mean = rate_mean / np.sum(rate_mean)
            if( len(np.where(rate_mean>=0.667)[0])==1 ) :
                this_emo = list(raters_cats_format)[np.where(rate_mean==np.max(rate_mean))[0][0]]
                this_clip_info.emotion_cat = this_emo
                array_of_clips = np.append(array_of_clips, this_clip_info)
    
    print("Total Clips", len(array_of_clips))
    
    if(len(array_of_clips)<1): raise Exception("No clips found")  
    return array_of_clips
                
    

def create_ShemoDB_file_objects(db_path, deselect=None):
    '''
    M = male speaker
    F = female speaker
    S = sadness
    A = anger
    H = happiness
    W = surprise
    F = fear
    N = neutral
    
    #https://github.com/pariajm/Persian-emotion_catal-Speech-Database-ShEMO
    '''

    print("Creating file info objects")

    Sex_format = {"M":"male\\", "F":"female\\"}

    emotion_cats_format = {'N':'N', 'H':'H', 'S':'S', 'A':'A', 'F':'F', 'U':'W'}

    pre_format = '*'
    post_format = '*.wav'

    array_of_clips = np.array([])

    for sex in Sex_format:
        for spkr_id in range(0, 60):
            speaker_id = str(spkr_id)
            if(spkr_id<10):
                speaker_id = '0' + str(spkr_id)

            for emotion_cat in emotion_cats_format:

                file_to_look = glob.glob(db_path +  Sex_format[sex]  + pre_format + speaker_id 
                    + emotion_cats_format[emotion_cat]  + post_format)
                
                for index, wav_file_found in enumerate(file_to_look):
                    this_clip_info = Clip_file_Class(4, wav_file_found, int(speaker_id), 3,  sex, emotion_cat, 1, None,None, None, None, statement=index, repetition=1)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    
                    #break    
                #break
        
    print("Total Clips", len(array_of_clips))
    return array_of_clips
    




def create_DEMOS_file_objects(db_path, deselect=['N']):
  
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

                file_to_look = glob.glob(db_path +  'DEMOS\\PR_' +  Sex_format[sex] + '_'+ speaker_id + '_' + Emotions_format[emotion]  + '*.wav')
                
                for index, wav_file_found in enumerate(file_to_look):
                    this_clip_info = Clip_file_Class(5, wav_file_found, int(speaker_id), 4, sex, emotion, 1, statement=index, repetition=1)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    #print(wav_file_found)

                file_to_look = glob.glob(db_path +  'NEU\\' +  Sex_format[sex] + '_'+ speaker_id + '_' + Emotions_format[emotion]  + '*.wav')
                
                for index, wav_file_found in enumerate(file_to_look):
                    this_clip_info = Clip_file_Class(5, wav_file_found, int(speaker_id), 4, sex, emotion, 1, statement=index, repetition=1)
                    array_of_clips = np.append(array_of_clips, this_clip_info)
                    #print(wav_file_found)


        
    print("Total Clips", len(array_of_clips))
    if(len(array_of_clips)<1): raise Exception("No clips found")  
    return array_of_clips




def create_DB_file_objects(db_name, db_path, deselect=None):
    '''
    Creates a list of file objected of all the wav files present in the db_path, as long as they are according to the formant of relevant 'db_name's original source

    return list of Clip_file_Class objects

    Object Contains:
    (string)filepath, (int)speaker_id, (int)scenario, (char)sex, (char)emotion_cat, (int)intensity_cat, (int)statement, (int)repetition, (int)db_id, (int)frame_count, (int)signal_len, (int)trimmed_len, (int)file_size
    
    Speech emotion_cat Recognition (SER) databases augmentation

    Currently supports these DBs

    db_name="EmoDB", db_path="C:\\DB\\EMO-DB\\wav\\";

    db_name="RAVDESS", db_path="C:\\DB\\RAVDESS\\";

    db_name="IEMOCAP", db_path="C:\\DB\\IEMOCAP_noVideo\\";

    db_name="ShemoDB", db_path="C:\\DB\\shemo\\";

    db_name="DEMoS", db_path="C:\\DB\\wav_DEMoS\\";
    
    db_name="MSPIMPROV", db_path="C:\\DB\\MSP-IMPROV\\";

    '''
    if(db_name=="EmoDB"):
        return create_EMODB_file_objects(db_path, deselect=deselect)

    elif(db_name=="RAVDESS"): 
        return create_RAVDESS_file_objects(db_path, deselect=deselect)
    elif(db_name=="IEMOCAP"): 
        return create_IEMOCAP_file_objects(db_path, deselect=deselect)
        
    elif(db_name=="ShemoDB"): 
        return create_ShemoDB_file_objects(db_path, deselect=deselect)
    elif(db_name=="DEMoS"): 
        return create_DEMOS_file_objects(db_path, deselect=deselect)
    elif(db_name=="MSPIMPROV"): 
        return create_MSPIMPROV_file_objects(db_path, deselect=deselect)
        
    
    else: raise Exception("Invalid DB name/key in db_sub_paths")



