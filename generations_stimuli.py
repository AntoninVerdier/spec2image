import os
import math
import numpy as np
from functions.Sample import Sound

# Generation of all wavfiles of interest
if not os.path.exists('Samples/'):
	os.makedirs('Samples/')

# Psychometrics task, Frequency discrimination
freqs = np.array(np.geomspace(12e3, 20e3, 16), dtype=np.int32)

for f in freqs:
	pure = Sound(samplerate=192000, amplitude=70)
	pure.simple_freq(f, duration=500)
	pure.save_wav(name='PT_{}Hz_{}ms_{}dB'.format(f, 500, 70), path='../Samples/')
