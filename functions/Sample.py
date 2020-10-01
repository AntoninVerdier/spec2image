import os
import numpy as np
import matplotlib.pyplot as plt 

from scipy.io import wavfile


class Sound():
	"""docstring for Sample

	Allow experimenter to generate tone paradigm
	"""
	def __init__(self):
		self.freq = []

	def simple_freq(self, frequency, duration=1000, samplerate=80000):
		"""Generate a pure tone signal for a given amount of time
		"""
		sample = duration * 0.001 * samplerate
		time = np.arange(sample)
		pure_tone = np.sin(2 * np.pi * frequency * int(frequency) * time / samplerate)

		wavfile.write(os.path.join('../Samples/', 'simple_freq.wav'), samplerate, pure_tone)
		

	def freq_modulation():
		pass


		
		