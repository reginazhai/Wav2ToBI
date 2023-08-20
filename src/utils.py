import os
import math
import numpy as np
from pydub import AudioSegment

# NOTE: current assumption for suffix is .brk or .ton. If this changes, change SUFFIX_LEN
SUFFIX_LEN = 3
PITCH_ACC = {
    'H*':1,
    'L+H*':2,
    'L*':3,
    'L*+H':4,
    'H+!H*': 5
}
ORIG_PITCH_MAP = {
    '!H*': 'H*',
    'L+!H*': 'L+H*',
    'L*+!H': 'L*+H'
}

# Splitting .wav files
# Code Splitting adapted from https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files @Shariful Islam Mubin

class SplitWavAudioMubin():
    def __init__(self, folder = None, filename = None):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        if not os.path.exists(self.folder + '/sliding/'):
            os.makedirs(self.folder + '/sliding/')
        split_audio.export(self.folder + '/sliding/' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split, window_size):
        total_mins = math.ceil(self.get_duration())
        print("Total Seconds:", total_mins)
        total_time = np.arange(0.0,total_mins, window_size)
        for tim in range(len(total_time)):
            i = total_time[tim]
            split_fn = str(tim) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print("Time:", str(i) + ' Done')
            if sec_per_split >= (total_mins - i):
                print('All splited successfully')
                break
        
def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False