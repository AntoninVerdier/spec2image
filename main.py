import os 
import scipy
import librosa
from librosa import load, display
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal as signal
from skimage.measure import block_reduce

import plot as pl 
import settings as sett

paths = sett.paths()
params = sett.parameters()

def fast_fourier(sample, samplerate):
	fft = scipy.fft(sample)

	return fft

def old_spectrogram(sample, samplerate, window_ms=20, stride_ms=10, max_freq=4000, eps=1e-14):
	# Test function, do not use
	window_size = int(window_ms * samplerate * 0.001)
	stride_size = int(stride_ms * samplerate * 0.001)

	# Truncate signal
	truncate_size = (len(sample) - window_size) % stride_size
	sample = sample[:len(sample) - truncate_size]

	nshape = (window_size, (len(sample) - window_size) // stride_size + 1)
	nstrides = (sample.strides[0], sample.strides[0] * stride_size)

	windows = np.lib.stride_tricks.as_strided(sample, shape=nshape, strides=nstrides)

	assert np.all(windows[:, 1] == sample[stride_size:(stride_size + window_size)])

	# Window weighting, square fft, scaling
	weighting = np.hanning(window_size)[:, None]

	fft = np.fft.rfft(windows * weighting, axis=0)
	fft = np.absolute(fft)
	fft = fft**2

	scale = np.sum(weighting**2) * samplerate
	fft[1:-1, :] *= (2.0 / scale)
	fft[(0, -1), :] /= scale 

	freqs = samplerate / window_size * np.arange(fft.shape[0])

	ind = np.where(freqs <= max_freq)[0][-1] + 1
	specgram = np.log(fft[:ind, :] + eps)

	return specgram


def spectro(sample, samplerate, window_ms=20, windows_ms=20, overlap=50):
	window_size = int(window_ms * samplerate * 0.001)
	overlap_size = overlap * 0.01* window_size

	spectrum, frequencies, times, im = plt.specgram(sample, Fs=samplerate, 
													NFFT=window_size, noverlap=overlap_size)

	plt.savefig(os.path.join(paths.path2Output, 'sample_spectrogram.png'))


	return spectrum, frequencies, times


def create_spatial_masks(size, res, random=False):
	if random:
		spatial_tensor = np.random.rand(size, size, res)
	else:
		spatial_tensor = np.zeros((size, size, res))

	return spatial_tensor


def create_amplitude_mask(res, random=False):
	if random:
		amplitude_mask = np.random.rand(res)
	else:
		amplitude_mask = np.ones(res)

	return amplitude_mask

def tonotopic_map(idx, tmap, freqs):
	return tmap * freqs[idx][:, np.newaxis, np.newaxis]

def downscale_tmaps(tmaps, block_size=(4, 4)):
	tmaps_reduced = []
	for i, tmap in enumerate(tonotopic_maps):
		tmap_reduced = block_reduce(tmap, block_size=block_size, func=np.mean)
		tmaps_reduced.append(tmap_reduced)

	return np.array(tmaps_reduced)




#Extract data
sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'example03.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))

# Reshape tonotopic maps to 625 x 500 arrays
tonotopic_maps = downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end
tonotopic_maps = tonotopic_maps[:-1, :, :]

# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:-len(sample)/2, 0])

# Visualize data through waveplot
pl.waveplot(sample, samplerate)

# Perform Fourier transform and plotting
# fft = fast_fourier(sample, samplerate)
# pl.fft(sample, samplerate, fft)

# Compute spectrogram
specgram, frequencies, times = spectro(sample, samplerate)
pl.spectrogram(specgram, frequencies, times)

# Create placeholder for spatial frequency masks and selectivity vector
amplitude_mask = create_amplitude_mask(params.freq_resolution, random=True)

# Extract frequencies for a given time
freq_series = [specgram[:, i] for i in range(specgram.shape[1])]

magnitude_indexs = [np.where(np.logical_and(frequencies >= params.freqs[i], frequencies < params.freqs[i+1])) 
					for i, fr in enumerate(params.freqs[:-1])]

downscaled_freqs = []
for i, freq in tqdm(enumerate(freq_series)):
	# Min max normalization of magnitude frequencies
	if np.max(freq) > 0:
		freq = (freq - np.min(freq))/(np.max(freq) - np.min(freq))

	downscaled_freqs.append([np.sum(freq[idx]) for idx in magnitude_indexs])
downscaled_freqs = np.array(downscaled_freqs)

# Create a generator since full array is too large
tonotopic_projections = np.array([tonotopic_maps * freq[:, np.newaxis, np.newaxis] for freq in downscaled_freqs])

pl.gif_projections(tonotopic_projections)


# pl.gif_projections(tonotopic_projections_gen)

# tonotopic_projection = tonotopic_maps * downscaled_freq[:, np.newaxis, np.newaxis]

# tonotopic_projections = np.array(tonotopic_projections)


# Apply frequency selectivity
# tensor_to_project = tensor_to_project * amplitude_mask[np.newaxis, np.newaxis, :]


