import numpy as np
import glob
import os
import SER_object

#http://emodb.bilderbar.info/docu/#docu






def create_EMODB_file_objects(db_path):
    #For RAVDESS dataset

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
                        this_clip_info = SER_object.Clip_file_Class(wav_file_found, int(speaker_id), 1,  Speakers_sex[speaker_id], emotion, 1, int(statement), int(repetition), db_id=2)
                        array_of_clips = np.append(array_of_clips, this_clip_info)
                        #print(wav_file_found)


    print("Total Clips", len(array_of_clips))
    return array_of_clips
    


