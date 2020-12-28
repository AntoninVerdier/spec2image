import os
from functions.Sample import Sound

# Generation of all wavfiles of interest
if not os.path.exists('Samples/'):
	os.makedirs('Samples/')

# Create pure frequency tone
pure = Sound()
pure.simple_freq(8000, duration=750)
pure.save_wav(name='Pure_tone_8k', path='Samples/pure_tone/')

pure2 = Sound()
pure2.simple_freq(16000, duration=750)
pure2.save_wav(name='Pure_tone_16k', path='Samples/pure_tone/')

# create pure frequency with noise
noise = Sound()
noise.freq_noise(16000, 0.1, duration=750)
noise.save_wav(name='Noise_0.1_16k', path='Samples/noise/')

noise2 = Sound()
noise2.freq_noise(16000, 0.8, duration=750)
noise2.save_wav(name='Noise_0.8_16k', path='Samples/noise/')

# create AM modulated tone
AM = Sound()
AM.amplitude_modulation(16000, 10, duration=750)
AM.save_wav(name='AM_10_16k', path='Samples/am/')

AM2 = Sound()
AM2.amplitude_modulation(16000, 100, duration=750)
AM2.save_wav(name='AM_100_16k', path='Samples/am/')

# create multi-frequencies sound
harm = Sound()
harm.multi_freqs([8000, 16000, 20000], duration=750)
harm.save_wav(name='Harmonics_8_16_20k', path='Samples/multif/')

# create frequency modulated tone
mod = Sound()
mod.freq_modulation(12000, 20000, duration=750)
mod.save_wav(name='Modul_12_20k', path='Samples/freq_mod/')

#Create harmonics
harmonics = Sound()
harmonics.harmonics(500, [0, 0, 1, 1, 0, 0], duration=2000)
harmonics.save_wav(name='Harmonics_500_001100', path='Samples/harmonics/')
