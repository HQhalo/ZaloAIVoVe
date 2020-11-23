import numpy as np
import os
import sys
from pydub import AudioSegment
from tqdm import tqdm
from pathlib import Path
import multiprocessing 
import pandas as pd
from Speaker import *
class Dataset:
    def __init__(self,pathDataset):
        self.pathDataset = pathDataset
        self.speakers = {}
    def scanFromList(self,unique_list):
      i= 0
      for line in unique_list:
        parts = line.split("/")
        if parts[-2] in self.speakers.keys():
          self.speakers[parts[-2]].addSoundFromPath(parts[-1])
        else:
          self.speakers[parts[-2]] = Speaker(self.pathDataset,parts[-2],i)
          i+=1
      return self.speakers

    def scan(self):
        i= 0
        for speakerName in os.listdir(self.pathDataset):
  
            if(os.path.isdir(os.path.join(self.pathDataset,speakerName))):
                speaker = Speaker(self.pathDataset,speakerName,i)
                i += 1
                speaker.scanSound()
                self.speakers[speakerName] = speaker
    
        return self.speakers
    
    def toFile(self,path):
      fTrain = open(os.path.join(path,"train.txt"),"w")
      fVal = open(os.path.join(path,"val.txt"),"w")

      for key in self.speakers:
        strTrain, strVal = self.speakers[key].trainValSet()
        fTrain.write(strTrain)
        fVal.write(strVal)
      fTrain.close()
      fVal.close()
    def getTestSet(self):
      audio = {"audio_1" : [], "audio_2" : [],"label" : []}
      #seft sets
      for speaker1 in tqdm(self.speakers.keys()):
        for speaker2 in self.speakers.keys():
          a1, a2 , l = self.speakers[speaker1].pair(self.speakers[speaker2])
          audio['audio_1'] += a1
          audio['audio_2'] += a2
          audio['label'] += l
      return audio
    def toString(self,path):
        f= open(os.path.join(path,"dataset.txt"),"w")
        for key in self.speakers.keys():
            f.write(self.speakers[key].getAllPathSound())
        f.close()
    
    def convertAllSoundToFlac(self,datasetMeta):
        f = open(datasetMeta,"r")
        lines = f.readlines()
        for line in tqdm(lines):
            line = line[0:-1]
            newline = line.replace("/content/VoveDataset/dataset","/content/VoveDataset/dataset_flac").split(".")[0]+".flac"
            Path(os.path.dirname(newline)).mkdir(parents=True, exist_ok=True)
            song = AudioSegment.from_wav(line)
            song.export(newline,format = "flac")
    def mixNoise(self):
        noises1 = Speaker("/content/musan/noise","free-sound",0)
        noises2 = Speaker("/content/musan/music","fma",0)
        noises1.scanSound(isNoise = True)
        noises2.scanSound(isNoise = True)
        soundNoises = noises1.sounds + noises2.sounds
        print(len(soundNoises))
        p = None
        for key in tqdm(self.speakers):
            p = multiprocessing.Process(target = self.speakers[key].mixNoise, args = (soundNoises,))
            p.start()
        p.join()
        print("Multi process stoped")
if __name__ == "__main__":
    data = Dataset("/content/VoveDataset/dataset")
    re = data.scan()
    # data.toString("/content")

    # data.convertAllSoundToFlac("/content/dataset.txt")

    # data.mixNoise()
    # data.scan()
    data.toFile("/content")


    # errTest = pd.read_csv("/content/train.txt")
    # unique_list = np.unique(np.array(errTest["path"]))
    # np.random.seed(42)
    # unique_list = np.random.choice(unique_list, 3000)
    # data.scanFromList(unique_list)

    # pd.DataFrame(data.getTestSet()).to_csv("/content/testSet.csv",index = False)

    