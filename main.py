import os 
import scipy
import librosa
from librosa import load, display
import numpy as np 
import matplotlib.pyplot as plt 

from scipy import signal as signal
from scipy.io import wavfile

import plot as pl 
import settings as sett

paths = sett.paths()
params = sett.parameters()

def fast_fourier(sample, samplerate):
	fft = scipy.fft(sample)

	return fft

def spectrogram(sample, samplerate, window_ms=20, stride_ms=10, max_freq=4000, eps=1e-14):
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


def spectro(sample, samplerate):
	frequencies, times, spectrogram = signal.spectrogram(sample, samplerate)

	# Perform min-max normalization
	spectrogram = (spectrogram - np.min(spectrogram))/(np.max(spectrogram) - np.min(spectrogram))

	return frequencies, times, spectrogram



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



#Extract data
data_path = '../Samples/'
sample, samplerate = librosa.load(os.path.join(data_path, 'example02.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:, 0])

# Visualize data through waveplot
pl.waveplot(sample, samplerate)

# Perform Fourier transform and plotting
fft = fast_fourier(sample, samplerate)
pl.fft(sample, samplerate, fft)

# Compute spectrogram
specgram = spectrogram(sample, samplerate)
#frequencies, times, spectrogram = spectro(sample, samplerate, plot=False)
pl.spectrogram(specgram)

# Create placeholder for spatial frequency masks and selectivity vector
spatial_tensor = create_spatial_masks(params.size_implant, params.freq_resolution, random=True)
amplitude_mask = create_amplitude_mask(params.freq_resolution, random=True)

# Extract frequencies for a given time
freq_series = [specgram[:, i] for i in range(specgram.shape[0])]

# Filter them by 2D mask
downscaled_freqs = []
for freq in freq_series:
	downscaled_freq = [np.sum((freq >= params.freqs[i-1]) & (freq < params.freqs[i])) for i, f in enumerate(params.freqs)]
	downscaled_freqs.append(downscaled_freq)

print(downscaled_freqs)
# masked_freq

# Apply frequency selectivity
# tensor_to_project = masked_freq * freq_selectivity[:, np.newaxis, np.newaxis]


