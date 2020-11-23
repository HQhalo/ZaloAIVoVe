import os
from Sound import *
from sklearn.model_selection import train_test_split
import random
import numpy as np
import itertools
NOSAMPLE = 2
class Speaker:
    def __init__(self,pathFolder, id, index):
        self.pathFolder = pathFolder
        self.id = id
        self.index = index
        self.sounds = []
        self.path = os.path.join(self.pathFolder,self.id)

    def scanSound(self,ext = "wav",isNoise = False):
        for filename in os.listdir(self.getPath()):
            if(os.path.isfile(os.path.join(self.path,filename)) and filename.split(".")[-1] == ext ):#and isNoise == False or ("noise" in filename and "RVB2014" in filename )):
                sound = Sound(self,filename)
                self.sounds.append(sound)
        self.sounds.sort(key = lambda x: x.filename)
    def addSoundFromPath(self,name):
      sound = Sound(self,name)
      self.sounds.append(sound)

    def getPath(self):
      return self.path
    def getAllPathSound(self):
      allPath = ''
      for sound in self.sounds:
        allPath +=sound.getPath()+"\n"
      return allPath
    def pair(self,speaker):
      audio1 = []
      audio2 = []
      l = []

      if self == speaker:
        for pairs in list(itertools.combinations(self.sounds, 2)):
          audio1 += [pairs[0].getPath()]
          audio2 += [pairs[1].getPath()]
          l += [1]
      else:
        random.seed(42)
        temp1 = random.sample(self.sounds,min(len(self.sounds),NOSAMPLE))
        random.seed(42)
        temp2 = random.sample(speaker.sounds,min(len(speaker.sounds),NOSAMPLE))
        
        for pairs in zip(temp1,temp2):
          audio1 += [pairs[0].getPath()]
          audio2 += [pairs[1].getPath()]
          l += [0]
      return [audio1,audio2,l]
    def trainValSet(self,test_size = 0.2):
      if self.index == 1:
        print(len(self.sounds))
      random.seed(42)
      suffle = random.sample(self.sounds,len(self.sounds))
      n = int(len(self.sounds)*test_size)
      train = suffle

      strStrain = ''
      strVal = ''
      for sound in train:
        strStrain += (sound.getPath() + "," + str(self.index))+"\n"
      return strStrain, strVal
    
    def toString(self):
      re = ""
      for sound in self.sounds:
        re += sound.getPath() + " " + str(self.index)+ "\n"
      return re

    def mixNoise(self,noiseList, ratio = 0.6,noise_delete = 0.7):
      random.seed(42)
      sampleSounds = random.sample(self.sounds,int(ratio*len(self.sounds)))
      np.random.seed(42)
      idxNoises = np.random.randint(len(noiseList),size = int(ratio*len(self.sounds)))
      temp = [True, False]
      np.random.seed(42)
      deteles = np.random.choice(temp,size = int(ratio*len(self.sounds)), p =[noise_delete,1-noise_delete])
      for sound, idxNoise, d in zip(sampleSounds,idxNoises,deteles):
        self.sounds.append(sound.mixNoise(noiseList[idxNoise],d))
      self.sounds.sort(key = lambda x: x.filename)