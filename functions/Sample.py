import os
import numpy as np
import matplotlib.pyplot as plt 

from scipy.io import wavfile

from tqdm import tqdm

class Sound():
	"""docstring for Sample

	Allow experimenter to generate tone paradigm
	"""
	def __init__(self, samplerate=80000):
		self.freq = []
		self.samplerate = samplerate

	def simple_freq(self, frequency, duration=1000):
		"""Generate a pure tone signal for a given amount of time
		"""
		sample = int(duration * 0.001) * self.samplerate
		time = np.arange(sample)
		pure_tone = np.sin(2 * np.pi * frequency * time / self.samplerate)

		wavfile.write(os.path.join('../Samples/', 'simple_freq.wav'), self.samplerate, pure_tone)

		return pure_tone

	def freq_modulation(self, start_freq, end_freq, duration=2500):
		"""Generate a pure tone signal for a given amount of time
		TODO : implemente class-wide function to add delay before and after stimulus
		"""
		sample = int(duration) * 0.001 * self.samplerate
		time = np.arange(sample)
		frequencies = np.linspace(start_freq, end_freq, sample)
		modulation = [np.sin(2 * np.pi * f * t / self.samplerate) for (f, t) in zip(np.linspace(start_freq, end_freq, sample), time)]

		wavfile.write(os.path.join('../Samples/', 'freq_modulation.wav'), self.samplerate, np.array(modulation))

		return modulation
		
	def amplitude_modulation(self, freq, am_freq, duration=5000):
		"""Generate an aplitude_modulated tone at a ref frequency
		"""
		sample = int(duration * 0.001) * self.samplerate
		time = np.arange(sample)
		amplitude = np.sin(2 * np.pi * am_freq * time / self.samplerate)
		modulated_signal = [A * np.sin(2* np.pi * freq * t / self.samplerate) for A, t in zip(amplitude, time)]
		
		wavfile.write(os.path.join('../Samples/', 'amplitude_modulation.wav'), self.samplerate, np.array(modulated_signal))


		return modulated_signal

	def freq_noise(self, freq, noise_vol, duration=2500):
		"""Create a pure tone with noise in the background of increasing intensity
		"""
		sample = int(duration * 0.001) * self.samplerate
		time = np.arange(sample)
		noise = np.array([noise_vol * np.random.random() for t in time])
		noisy_signal = noise + np.array([np.sin(2 * np.pi * freq * t / self.samplerate) for t in time])

		wavfile.write(os.path.join('../Samples/', 'freq_noise.wav'), self.samplerate, noisy_signal)

		return noisy_signal

	def multi_freqs(self, freqs, duration=2500):
		""" used to create harmonics
		"""
		sample = int(duration * 0.001) * self.samplerate
		time = np.arange(sample)
		harmonics = np.sum(np.array([[np.sin(2 * np.pi * freq * t / self.samplerate) for t in time] for freq in freqs]), axis=0)

		wavfile.write(os.path.join('../Samples/', 'freq_noise.wav'), self.samplerate, harmonics)

		return harmonics
