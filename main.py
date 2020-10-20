import os 
import math
import scipy
import argparse
import librosa
from librosa import load, display
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm import tqdm
from scipy.io import wavfile
from skimage.measure import block_reduce

import plot as pl 
import settings as sett
import processing as proc
from functions.Sample import Sound

paths = sett.paths()
params = sett.parameters()

parser = argparse.ArgumentParser(description='Parameters for computing')

parser.add_argument('--rectangle', '-r', type=int, default=None, 
				    help='Number of stimulation rectangles')

args = parser.parse_args()



plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['figure.dpi'] 

#Extract data
sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'pipeline_test.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))
tonotopic_maps = proc.downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end and white noise at the beginning
tonotopic_maps = tonotopic_maps[1:-1, :, :]

# Normalization of tonotopic maps and inversion (bright spots should be of interest)
for i, tmap in enumerate(tonotopic_maps):
	tonotopic_maps[i] = (tmap - np.min(tmap))/(np.max(tmap) - np.min(tmap))
	tonotopic_maps[i] = 1 - tmap 


# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:-len(sample)/2, 0])

# Compute spectrogram
specgram, frequencies, times = proc.spectro(sample, samplerate)

# Compute rectangle stimulation if needed
if args.rectangle is not None:
	rect_stim, weighted_tmap, min_4, min_32 = proc.rectangle_stim(tonotopic_maps[0], tonotopic_maps[2], 5)

	# Special cas ewhere tonotopic maps are 4 and 32
	freqs_bounds = np.linspace(4e3, 32e3, num=args.rectangle + 1)
	freq_series = [specgram[:, i] for i in range(specgram.shape[1])]


acti_rect = []
for freq in freq_series:
	# !!!!!!!!!!!!!!!!! It needs to be frequencies not amplitudes
	freq = (freq - np.min(freq)) / (np.max(freq) - np.min(freq))
	print(freq)
	acti_rect.append([np.sum(freq[(freq >= freqs_bounds[i-1]) & (freq < f)]) for i, f in enumerate(freqs_bounds[1:])])
acti_rect = np.array(acti_rect)
print(acti_rect.shape)

plt.close()
plt.imshow(acti_rect)
plt.show()


# Extract frequencies for a given time
# magnitudes = proc.gaussian_windowing(specgram, frequencies)
# tonotopic_projections = np.array([tonotopic_maps * mag[:, np.newaxis, np.newaxis] for mag in magnitudes])

# Downscale projection to match implants' characteristics


plt.scatter(min_4[0], min_4[1], marker='o', c='red')
plt.scatter(min_32[0], min_32[1], marker='o', c='red')
plt.plot([min_4[0], min_32[0]], [min_4[1], min_32[1]])

for i, rect in enumerate(rect_stim):
	weighted_tmap[rect[:, 1], rect[:, 0]] = 0.2 * i
	plt.imshow(weighted_tmap, cmap='coolwarm')
plt.title('Stimulation along a tonotopic axis')
plt.savefig('update.png')
plt.show() 

# stimulus = []
# for i, rect in enumerate(rect_stim):
# 	buffer_map = np.copy(weighted_tmap) # relpace function copy by zero_like for mask
# 	buffer_map[rect[:, 0], rect[:, 1]] = 1
# 	stimulus.append([buffer_map for i in range(1000)])
# stimulus = np.concatenate(stimulus, axis=0)

	# Create stimulus across time


projections = proc.implant_projection(stimulus)
#Script for checking


#plt.imshow(projections[1001], cmap='coolwarm')

#plt.show()
