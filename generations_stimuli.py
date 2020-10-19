import os
from functions.Sample import Sound


# Generation of all wavfiles of interest
if not os.path.exists('Samples/'):
	os.makedirs('Samples/')

# create pure frequency tone
pure = Sound()
pure.simple_freq(16000, duration=1000)
pure.save_wav(name='Pure_tone_16k')

# create pure frequency with noise
pure = Sound()
pure.freq_noise(16000, 0.3, duration=1000)
pure.save_wav(name='Noise_0.3_16k')

# create am modulated tone
pure = Sound()
pure.amplitude_modulation(16000, 100, duration=1000)
pure.save_wav(name='AM_100_16k')

# create am modulated tone
pure = Sound()
pure.amplitude_modulation(16000, 50, duration=1000)
pure.save_wav(name='AM_50_16k')

# create harmonics
pure = Sound()
pure.multi_freqs([8000, 16000, 20000], duration=1000)
pure.save_wav(name='Harmonics_8_16_20k')

# create pure frequency tone
pure = Sound()
pure.freq_modulation(12000, 20000, duration = 1000)
pure.save_wav(name='Modul_12_20k')

