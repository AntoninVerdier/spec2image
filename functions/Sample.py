import os
import numpy as np
import matplotlib.pyplot as plt 

from scipy.io import wavfile

from tqdm import tqdm

class Sound():
	""" Class for creating sound paradigm with multiple features

	Attributes 
	----------
	name : str
		Sample's name. Will be used for saving
	signal : array
		Sound signal
	freq : dict 
		Dictionnary containing frequencies information for quick access. Not yet supported with __add__ method. 
	samplerate : int
		Number of samples needed in the isgnal per unit of time

	Constructors
	------------
	__init__(self, samplerate=80000)
		Initialize object and attribute
	__add__(self, other)
		Allow user to add two Sound's signals to create a more complex Sound object

	Methods 
	-------
	delay(self, duration)
		Generate a silence for a given duration
	simple_freq(self, frequency, duration=1000)
		Generate a pure tone signal for a given duration
	freq_modulation(self, start_freq, end_freq, duration=2500)
		 a signal of increase or decreasing frequencies
	amplitude_modulation(self, freq, am_freq, duration=5000)
		Generate an aplitude-modulated tone at a reference frequency
	freq_noise(self, freq, noise_vol, duration=2500)
		Create a pure tone with noise in the background of increasing intensity
	multi_freqs(self, freqs, duration=2500)
		Generate multiple frequency harmonics
	save_wav(self, name=None):
		Save the signal as a .wav file

	"""
	def __init__(self, samplerate=80000):
		self.name = 'test'
		self.signal = None
		self.freq = None
		self.samplerate = samplerate

	def __add__(self, other):
		"""Define how to assemble generated sounds
		"""
		assert self.samplerate == other.samplerate, 'Signals must have the same samplerate'
		assert (self.signal is not None) & (other.signal is not None), 'Signals must be defined' 

		newSound = Sound(samplerate=self.samplerate)
		newSound.signal = np.concatenate((self.signal, other.signal))

		return newSound

	def delay(self, duration):
		""" Generate a silence for a given duration

		Parameters
		----------
		duration : int
			Duration of the delay in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		self.signal = np.array(np.zeros(sample))
		print(sample)

		# return self.signal

	def simple_freq(self, frequency, duration=1000):
		"""Generate a pure tone signal for a given duration

		Parameters
		----------
		frequency : int
			Frequency of the pure tone to generate
		duration : int, optional
			Duration of the tone in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		pure_tone = np.sin(2 * np.pi * frequency * time / self.samplerate)

		self.signal = np.array(pure_tone)
		self.freq = {'simple' : frequency}

		# return pure_tone

	def freq_modulation(self, start_freq, end_freq, duration=2500):
		"""Generate a signal of increase or decreasing frequencies

		Parameters
		----------
		start_freq
			Starting frequency of the signal
		end_freq
			Ending frequency of the signal 
		duration : int, optional
			Duration of the sound sample in ms

		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		frequencies = np.linspace(start_freq, end_freq, sample)
		modulation = [np.sin(2 * np.pi * f * t / self.samplerate) for (f, t) in zip(np.linspace(start_freq, end_freq, sample), time)] # or log add option

		self.signal = np.array(modulation)
		self.freq = {'start_freq': start_freq, 'end_freq': end_freq}


		# return modulation
		
	def amplitude_modulation(self, freq, am_freq, duration=5000):
		"""Generate an aplitude-modulated tone at a reference frequency

		Parameters
		----------
		freq : int
			Frequency of the signal
		am_freq : int
			Frequency of the amplitude-modulation
		duration : int
			Duration of the soudn sample in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		amplitude = np.sin(2 * np.pi * am_freq * time / self.samplerate)
		modulated_signal = [A * np.sin(2* np.pi * freq * t / self.samplerate) for A, t in zip(amplitude, time)]
		
		self.signal = np.array(modulated_signal)
		self.freq = {'freq': frequency, 'am_freq': am_freq}

		# return modulated_signal

	def freq_noise(self, freq, noise_vol, duration=2500):
		"""Create a pure tone with noise in the background of increasing intensity

		Parameters
		----------
		freq : int
			Frequency of the signal
		noise_vol : float
			Volume of the white noise in background
		duration : int, optional
			Duration of the sound sample in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		noise = noise_vol * np.random.normal(0, 1, len(time))
		noisy_signal = noise + np.array([np.sin(2 * np.pi * freq * t / self.samplerate) for t in time])

		self.signal = np.array(noisy_signal)
		self.freq = {'freq': freq, 'noise_vol': noise_vol}

		# return noisy_signal

	def multi_freqs(self, freqs, duration=2500):
		""" Generate multiple frequency harmonics

		Parameters
		----------
		freqs : list
			Frequencies of the signal
		duration : int, optional

		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		harmonics = np.sum(np.array([[np.sin(2 * np.pi * freq * t / self.samplerate) for t in time] for freq in freqs]), axis=0)

		self.signal = np.array(harmonics)
		self.freq = {'freq{}'.format(i): f for i, f in enumerate(freqs)}

		# return harmonics

	def save_wav(self, name=None):
		""" Save the signal as a .wav file

		Parameters 
		----------
		name : str, optional
			Name fo the file to save
		"""
		if name is None:
			name = self.name

		assert self.signal is not None, 'You must define a signal to save'
		wavfile.write(os.path.join('../Samples/{}.wav'.format(name)), self.samplerate, self.signal)


