import os 
import scipy
import librosa
from librosa import load, display
import numpy as np 
import matplotlib.pyplot as plt 

from scipy import signal as signal
from scipy.io import wavfile

import settings as sett

paths = sett.paths()
params = sett.parameters()

#Extract data
data_path = '../Samples/'
sample, samplerate = librosa.load(os.path.join(data_path, 'example02.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

print(sample.strides[0])

print(len(sample))
# plt.figure()
# display.waveplot(y=sample, sr=samplerate)
# plt.xlabel('Time (sec)')
# plt.ylabel('Amplitude')
# plt.show()
# Remove stereoif needed
# sample = np.transpose(sample[:, 0])

def fast_fourier(sample, samplerate, plot=False):
	fft = scipy.fft(sample)
	if plot:
		n = len(sample)
		T = 1/samplerate
		xf = np.linspace(0, 1/(2*T), n//2)
		plt.plot(xf, 2/n * np.abs(fft[:n//2]))
		plt.xlabel('Frequency')
		plt.ylabel('Magnitude')
		plt.show()

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




def spectro(sample, samplerate, plot=False):
	frequencies, times, spectrogram = signal.spectrogram(sample, samplerate)

	# Perform min-max normalization
	spectrogram = (spectrogram - np.min(spectrogram))/(np.max(spectrogram) - np.min(spectrogram))

	if plot:
		# To save in desired location
		plt.pcolormesh(times, frequencies, spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()

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


specgram = spectrogram(sample, samplerate)
plt.pcolormesh(specgram)
plt.show()

frequencies, times, spectrogram = spectro(sample, samplerate, plot=False)
print(spectrogram.shape)


spatial_tensor = create_spatial_masks(params.size_implant, params.freq_resolution, random=True)
amplitude_mask = create_amplitude_mask(params.freq_resolution, random=True)

# Extract frequencies for a given time
freq_series = [spectrogram[:, i] for i in range(spectrogram.shape[0])]

# Filter them by 2D mask

# Apply frequency selectivity



