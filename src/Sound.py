import os
from pydub import AudioSegment
import random
import librosa
class Sound:
    def __init__(self,speaker,filename):
        self.filename = filename
        self.speaker = speaker
    
    def getPath(self):
        return os.path.join(self.speaker.getPath(),self.filename)
    def getNameSound(self):
        return self.filename.split(".")[0]
    def mixNoise(self,noiseSound,delete= False):

        sourceSoundData = AudioSegment.from_file(self.getPath())
       
        noiceSoundData = AudioSegment.from_file(noiseSound.getPath()) - 15
        if sourceSoundData.channels > 1:       
            sourceSoundData = sourceSoundData.set_channels(1)

        newSoundData = sourceSoundData.overlay(noiceSoundData)

        newSoundData.export(self.speaker.getPath()+"/" + self.getNameSound()+"_"+noiseSound.getNameSound()+".wav", format = "wav")
        if delete == True and os.path.exists(self.getPath()):
          os.remove(self.getPath())
        newsound = Sound(self.speaker,  self.getNameSound()+"_"+noiseSound.getNameSound()+".wav")
        return newsound
