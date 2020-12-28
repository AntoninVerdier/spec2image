import os
import argparse
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

		""" Constructor at initialization
		"""
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

		# return self.signal

	def simple_freq(self, frequency, duration=500, phase=0):
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

	def freq_modulation(self, start_freq, end_freq, duration=500):
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
		time = np.linspace(0, duration * 0.001, num=sample)
		#frequencies =  np.linspace(start_freq, end_freq / 2, num=sample)

		k = (end_freq - start_freq)/ (duration*0.001)
		sweep = (start_freq + k/2 * time) * time
		modulation = np.sin(2* np.pi * sweep)

		# modulation = signal.chirp(t, f0=4000, f1=16000, t1=10, method='linear')

		# modulation = np.sin(2 * np.pi * frequencies * time)

		
		self.signal = np.array(modulation)
		self.freq = {'start_freq': start_freq, 'end_freq': end_freq}


		# return modulation
		
	def amplitude_modulation(self, freq, am_freq, duration=500):
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
		self.freq = {'freq': freq, 'am_freq': am_freq}

		# return modulated_signal

	def freq_noise(self, freq, noise_vol, duration=500):
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

	def multi_freqs(self, freqs, duration=500):
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

	def save_wav(self, path=None, name=None):
		""" Save the signal as a .wav file

		Parameters 
		----------
		name : str, optional
			Name fo the file to save
		"""
		assert self.signal is not None, 'You must define a signal to save'

		if name is None:
			name = self.name

		if path is None:
			wavfile.write(os.path.join('../Samples/{}.wav'.format(name)), self.samplerate, self.signal)
		else:
			wavfile.write(os.path.join(path, name + '.wav'), self.samplerate, self.signal)

def main():
	parser = argparse.ArgumentParser(description='Parameters for computing')

	parser.add_argument('--inline', '-i', action='store_true', 
					    help='for inline generated sounds')
	parser.add_argument('--puretone', '-p', type=int,
						help='Generate a pure tone frequency, please specify frequency in Hz')
	parser.add_argument('--noise', '-n', type=float, nargs=2,
						help='Specify frequency (Hz) and noise(btw 0 and 1')
	parser.add_argument('--ampmod', '-am', type=int, nargs=2, 
						help='Amplitude modulation. Base frequency and modulation frequency in Hz')
	parser.add_argument('--harmonic', '-ha', type=int, nargs='*',
	 					help='Generate harmnics Enter frequencies in Hz')
	parser.add_argument('--freqmod', '-fm', type=int, nargs=2, 
						help='Ramp frequency generation')
	parser.add_argument('--duration', '-d', type=int, default=500,
						help='Duration of the stimulus in ms')
	parser.add_argument('--path', '-a', type=str, default='Samples/',
						help='Path where to save produced stimulus')
	parser.add_argument('--name', '-na', type=str, default=None, 
						help='Name of the file generated')

	args = parser.parse_args()

	if args.inline:
		if not os.path.exists(args.path):
			os.makedirs(args.path)

		if args.puretone:
			pure = Sound()
			pure.simple_freq(args.puretone, duration=args.duration)
			pure.save_wav(path=args.path, name=args.name)
		
		elif args.noise:
			noise = Sound()
			noise.freq_noise(args.noise[0], args.noise[1], duration=args.duration)
			noise.save_wav(path=args.path, name=args.name)

		elif args.ampmod:
			am = Sound()
			am.amplitude_modulation(args.ampmod[0], args.ampmod[1], duration=args.duration)
			am.save_wav(path=args.path, name=args.name)

		elif args.freqmod:
			freqmod = Sound()
			freqmod.freq_modulation(args.freqmod[0], args.freqmod[1], duration=args.duration)
			freqmod.save_wav(path=args.path, name=args.name)

		elif args.harmonic:
			harmonic = Sound()
			harmonic.multi_freqs(args.harmonic, duration=args.duration)
			harmonic.save_wav(path=args.path, name=args.name)

if __name__=="__main__":
	main()
