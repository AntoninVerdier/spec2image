import os 
import math
import scipy
import librosa
from librosa import load, display
import numpy as np 
import matplotlib.pyplot as plt 

from tones import SINE_WAVE
from tones.mixer import Mixer 

from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal as signal
from skimage.measure import block_reduce

import plot as pl 
import settings as sett
import processing as proc
from functions.Sample import Sound

paths = sett.paths()
params = sett.parameters()

plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['figure.dpi'] 

# Generate sample pipeline
delay1 = Sound()
delay1.delay(1000)

stimulus1 = Sound()
stimulus1.freq_noise(1000, 0.8, duration=3000)

delay2 = Sound()
delay2.delay(500)

pipeline = delay1 + stimulus1 + delay2
pipeline.name = 'pipeline_test'
pipeline.save_wav()

#Extract data
sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'pipeline_test.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))

# Reshape tonotopic maps to 625 x 500 arrays for computation purposes
#tonotopic_maps = proc.downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end and white noise at the beginning
tonotopic_maps = tonotopic_maps[1:-1, :, :]

# Normalization of tonotopic maps and inversion (bright spots should be of interest)
for i, tmap in enumerate(tonotopic_maps):
	tonotopic_maps[i] = (tmap - np.min(tmap))/(np.max(tmap) - np.min(tmap))
	tonotopic_maps[i] = 1 - tmap


# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:-len(sample)/2, 0])

# Visualize data through waveplot
pl.waveplot(sample, samplerate)

# Perform Fourier transform and plotting
# fft = fast_fourier(sample, samplerate)
# pl.fft(sample, samplerate, fft)

# Compute spectrogram
specgram, frequencies, times = proc.spectro(sample, samplerate)
# pl.spectrogram(specgram, frequencies, times)

# Extract frequencies for a given time
freq_series = [specgram[:, i] for i in range(specgram.shape[1])]

# Define windows for gaussian windowing
gaussian_windows = [signal.gaussian(math.log(f/1000, 2)*1000, math.log(f/1000, 2)*150) for f in params.freqs]

# Get indices of frequencies of interest
freq_idxs = [np.where(np.logical_and(frequencies >= params.freqs[i]-len(win)/2, frequencies < params.freqs[i]+len(win)/2)) 
					for i, win in enumerate(gaussian_windows)]
win_idxs = [np.array(frequencies[idx] - np.min(frequencies[idx])).astype(int) for idx in freq_idxs]

magnitudes = []
for i, freq in enumerate(freq_series):
	magnitudes.append([np.sum(freq[freq_idxs[i]] * win[win_idxs[i]]) for i, win in enumerate(gaussian_windows)])

magnitudes = np.array(magnitudes)
magnitudes = magnitudes/np.max(magnitudes)

# Multiply magnitudes with tonotopic maps
tonotopic_projections = np.array([tonotopic_maps * mag[:, np.newaxis, np.newaxis] for mag in magnitudes])

# Downscale projection to match implants' characteristics

#pl.figure_1(projections, tonotopic_projections, specgram, sample, samplerate, 20, 50)
# pl.gif_projections(tonotopic_projections

rect_stim, weighted_tmap, min_4, min_32 = proc.rectangle_stim(tonotopic_maps[0], tonotopic_maps[2], 5)

plt.scatter(min_4[0], min_4[1], marker='o', c='red')
plt.scatter(min_32[0], min_32[1], marker='o', c='red')
plt.plot([min_4[0], min_32[0]], [min_4[1], min_32[1]])

for i, rect in enumerate(rect_stim):
	weighted_tmap[rect[:, 1], rect[:, 0]] = 0.2 * i
	plt.imshow(weighted_tmap, cmap='coolwarm')
plt.title('Stimulation along a tonotopic axis')
plt.savefig('update.png')
plt.show()

stimulus = []
for i, rect in enumerate(rect_stim):
	buffer_map = np.copy(weighted_tmap) # relpace function copy by zero_like for mask
	buffer_map[rect[:, 0], rect[:, 1]] = 1
	stimulus.append([buffer_map for i in range(1000)])
stimulus = np.concatenate(stimulus, axis=0)

	# Create stimulus across time


projections = proc.implant_projection(stimulus)
#Script for checking


#plt.imshow(projections[1001], cmap='coolwarm')

#plt.show()
