import os
import numpy as np
import matplotlib.pyplot as plt 

from scipy.io import wavfile

from tqdm import tqdm

class Sound():
	"""docstring for Sample

	Allow experimenter to generate tone paradigm
	"""
	def __init__(self, samplerate= 80000):
		self.freq = []
		self.samplerate = samplerate

	def simple_freq(self, frequency, duration=1000, samplerate=80000):
		"""Generate a pure tone signal for a given amount of time
		"""
		sample = int(duration * 0.001) * samplerate
		time = np.arange(sample)
		pure_tone = np.sin(2 * np.pi * frequency * time / samplerate)

		wavfile.write(os.path.join('../Samples/', 'simple_freq.wav'), samplerate, pure_tone)

		return pure_tone

	def freq_modulation(self, start_freq, end_freq, duration=2500, samplerate=80000):
		"""Generate a pure tone signal for a given amount of time
		TODO : implemente class-wide function to add delay before and after stimulus
		"""
		sample = int(duration) * 0.001 * samplerate
		time = np.arange(sample)
		frequencies = np.linspace(start_freq, end_freq, sample)
		modulation = [np.sin(2 * np.pi * f * t / samplerate) for (f, t) in zip(np.linspace(start_freq, end_freq, sample), time)]

		wavfile.write(os.path.join('../Samples/', 'freq_modulation.wav'), samplerate, np.array(modulation))

		return modulation
		
	def amplitude_modulation(self, freq, am_freq, duration=5000, samplerate=80000):
		"""Generate an aplitude_modulated tone at a ref frequency
		"""
		sample = int(duration * 0.001) * samplerate
		time = np.arange(sample)
		amplitude = np.sin(2 * np.pi * am_freq * time / samplerate)
		modulated_signal = [A * np.sin(2* np.pi * freq * t / samplerate) for A, t in zip(amplitude, time)]
		
		wavfile.write(os.path.join('../Samples/', 'amplitude_modulation.wav'), samplerate, np.array(modulated_signal))


		return modulated_signal

	def freq_noise(self, freq, noise_vol, duration=2500, samplerate=80000):
		"""Create a pure tone with noise in the background of increasing intensity
		"""
		sample = int(duration * 0.001) * samplerate
		time = np.arange(sample)
		noise = np.array([noise_vol * np.random.random() for t in time])
		noisy_signal = noise + np.array([np.sin(2 * np.pi * freq * t / samplerate) for t in time])

		wavfile.write(os.path.join('../Samples/', 'freq_noise.wav'), samplerate, noisy_signal)

		return noisy_signal