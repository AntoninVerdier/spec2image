import os
import math
import numpy as np
from functions.Sample import Sound

# Generation of all wavfiles of interest
if not os.path.exists('Samples/'):
	os.makedirs('Samples/')

# Psychometrics task, Frequency discrimination
ams = np.array(np.geomspace(20, 200, 16), dtype=np.int32)
print(ams)

for a in ams:
	pure = Sound(samplerate=192000, amplitude=70)
	pure.amplitude_modulation(10e3, a, duration=500)
	pure.save_wav(name='PT_{}Hz_{}ms_{}dB'.format(a, 500, 70), path='../Samples/Samples_AM_20_200_New/')
