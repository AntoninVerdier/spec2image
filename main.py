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
from functions import Sample as samp

paths = sett.paths()
params = sett.parameters()
Sound = samp.Sound()

Sound.simple_freq(440)


def create_sine_up(time_s=10):
	time = np.arange()


def create_pure_tone():
	mixer = Mixer(80000, 0.8)
	mixer.create_track(0, SINE_WAVE, attack=0.01, decay=0.1)
	mixer.add_note(0, note='a', octave=9, duration=6, endnote='g')
	mixer.write_wav(os.path.join(paths.path2Sample, 'tones.wav'))


def fast_fourier(sample, samplerate):
	fft = scipy.fft(sample)

	return fft

def implant_projection(tmaps):
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

	plt.savefig(os.path.join(paths.path2Output, 'sample_spectrogram.png'))


	return spectrum, frequencies, times


def tonotopic_map(idx, tmap, freqs):
	return tmap * freqs[idx][:, np.newaxis, np.newaxis]

def downscale_tmaps(tmaps, block_size=(4, 4)):
	tmaps_reduced = []
	for i, tmap in enumerate(tonotopic_maps):
		tmap_reduced = block_reduce(tmap, block_size=block_size, func=np.mean)
		tmaps_reduced.append(tmap_reduced)

	return np.array(tmaps_reduced)

#Extract data
sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'tones.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))

# Reshape tonotopic maps to 625 x 500 arrays
tonotopic_maps = downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end
tonotopic_maps = tonotopic_maps[1:-1, :, :]

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

windows = [signal.gaussian(math.log(f/1000, 2)*1000, math.log(f/1000, 2)*150) for f in params.freqs]

idxs = [np.where(np.logical_and(frequencies >= params.freqs[i]-len(win)/2, frequencies < params.freqs[i]+len(win)/2)) 
					for i, win in enumerate(windows)]

win_idxs = [np.array(frequencies[idx] - np.min(frequencies[idx])).astype(int) for idx in idxs]

downscaled_freqs = []
for i, freq in enumerate(freq_series):
	# Min max normalization of magnitude frequencies
	#if np.max(freq) > 0:
	#	freq = (freq - np.min(freq))/(np.max(freq) - np.min(freq))

	downscaled_freqs.append([np.sum(freq[idxs[i]] * win[win_idxs[i]]) for i, win in enumerate(windows)])

downscaled_freqs = np.array(downscaled_freqs)
downscaled_freqs = downscaled_freqs/np.max(downscaled_freqs)

# Create a generator since full array is too large
tonotopic_projections = np.array([tonotopic_maps * freq[:, np.newaxis, np.newaxis] for freq in downscaled_freqs])
tmaps = implant_projection(tonotopic_projections)

pl.figure_1(tmaps, spectro, sample, samplerate, 20, 50)

# pl.gif_projections(tonotopic_projections)


