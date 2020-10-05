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
from functions.Sample import Sound

paths = sett.paths()
params = sett.parameters()


def fast_fourier(sample, samplerate):
	""" Compute a Fast Fourier Transform for visualization purposes

	Parameters
	----------
	sample : array
		Array containaing the raw signal
	samplerate : int
		Number of sample in the signal per unit of time

	Returns
	-------
	array
		Fast-Fourier Transform of the signal

	"""
	fft = scipy.fft(sample)

	return fft

def implant_projection(tmaps):
	""" Downscale the final cortical stimulation to match capacitty of the implant

	Parameters
	----------
	tmaps : array
		Images to project to the cortex. Shape must be (time_points, frequencies, width, height)

	Returns
	-------
	array
		Images downsclaed to send to implant. Shape is (time_point, width, height)

	"""
	# Average stimulation pattern over frequencies to get weighted map
	tmaps = np.mean(tmaps, axis=1)
	
	width_cut = tmaps.shape[1] % params.size_implant
	height_cut = tmaps.shape[2] % params.size_implant

	# Cut excess borders
	tmaps = tmaps[:, width_cut:, height_cut:]

	tmap_implant= block_reduce(tmaps, block_size=(1, tmaps.shape[1] // params.size_implant, tmaps.shape[2] // params.size_implant), func=np.mean)

	return tmap_implant

def spectro(sample, samplerate, window_ms=20, windows_ms=20, overlap=50):
	window_size = int(window_ms * samplerate * 0.001)
	overlap_size = overlap * 0.01* window_size

	spectrum, frequencies, times, im = plt.specgram(sample, Fs=samplerate, 
													NFFT=window_size, noverlap=overlap_size)


	plt.title('Spectrogram of sound sample')
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')

	plt.savefig(os.path.join(paths.path2Output, 'sample_spectrogram.png'))


	return spectrum, frequencies, times

def downscale_tmaps(tmaps, block_size=(4, 4)):
	tmaps_reduced = []
	for i, tmap in enumerate(tonotopic_maps):
		tmap_reduced = block_reduce(tmap, block_size=block_size, func=np.mean)
		tmaps_reduced.append(tmap_reduced)

	return np.array(tmaps_reduced)

# Generate sample pipeline
delay1 = Sound()
delay1.delay(1000)

stimulus1 = Sound()
stimulus1.multi_freqs([110, 195, 329], duration=3000)

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
tonotopic_maps = downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end and white noise at the beginning
tonotopic_maps = tonotopic_maps[1:-1, :, :]

# Normalization of tonotopic maps and inversion (bright spots should be of interest)
for i, tmap in enumerate(tonotopic_maps):
	tonotopic_maps[i] = (tmap -np.min(tmap))/(np.max(tmap) - np.min(tmap))
	tonotopic_maps[i] = 1 - tonotopic_maps[i]


# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:-len(sample)/2, 0])

# Visualize data through waveplot
# pl.waveplot(sample, samplerate)

# Perform Fourier transform and plotting
# fft = fast_fourier(sample, samplerate)
# pl.fft(sample, samplerate, fft)

# Compute spectrogram
specgram, frequencies, times = spectro(sample, samplerate)
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
projections = implant_projection(tonotopic_projections)

pl.figure_1(projection, tonotopic_projections, spectro, sample, samplerate, 20, 50)
# pl.gif_projections(tonotopic_projections)


