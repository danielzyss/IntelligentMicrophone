import numpy as np
import os
import scipy.io.wavfile
import scipy.interpolate
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import random

dir = os.getcwd()
os.chdir('Data')
men_file = os.listdir('men')
women_file = os.listdir('women')
vowels = ['uw', 'ae', 'ah', 'aw', 'eh', 'ei', 'er', 'ih', 'iy', 'oa', 'oo', 'uh']

sounds = []
for v in vowels:
    sounds.append([])

for i, f in enumerate(men_file):
    for j , v in enumerate(vowels):
        if v in f:
            r, sound = scipy.io.wavfile.read('men/' + f)
            os.chdir(dir)
            plt.plot(sound)
            plt.xlabel('time')
            plt.ylabel('amplitude')
            plt.title('original signal')
            plt.savefig('figures/originalinput.png')
            plt.close()
            sound = sound/np.max(np.abs(sound), axis=0)
            plt.plot(sound)
            plt.xlabel('time')
            plt.ylabel('amplitude')
            plt.title('pre-processed signal')
            plt.savefig('figures/processedinput.png')
            plt.close()
            sys.exit()
            sounds[j].append(sound)

for i, f in enumerate(women_file):
    for j , v in enumerate(vowels):
        if v in f:
            r, sound = scipy.io.wavfile.read('women/' + f)
            sound = sound/np.max(np.abs(sound), axis=0)
            sounds[j].append(sound.astype(np.float64))

sounds = np.array(sounds)
idx = random.sample(range(0, sounds.shape[1]), sounds.shape[1])
X = sounds[:,idx]
Y = np.repeat(np.array(vowels), X.shape[1]).reshape((len(vowels), X.shape[1])).astype(np.str)

np.save('voweldataset/sounds.npy', X)
np.save('voweldataset/target.npy', Y)




print('DATASET EXPORTED TO \'voweldataset\' DIRECTORY')