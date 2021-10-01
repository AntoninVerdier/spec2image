import os
import librosa
import numpy as np

from librosa import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

from tqdm import tqdm

import settings as sett
paths = sett.paths()

def waveplot(sample, samplerate):
	""" Create a waveplot figure of the raw sample
	"""
	plt.figure(figsize=(12, 6))

	display.waveplot(y=sample, sr=samplerate)

	plt.title('Waveplot of sound sample')
	plt.xlabel('Time (sec)')
	plt.ylabel('Amplitude')

	plt.savefig(os.path.join(paths.path2Output, 'sample_waveplot.png'))
	plt.close()


def fft(sample, samplerate, fft):
	n = len(sample)
	T = 1/samplerate
	xf = np.linspace(0, 1/(2*T), n//2)
	yf = 2/n * np.abs(fft[:n//2])

	plt.plot(xf, yf)

	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude')

	plt.savefig(os.path.join(paths.path2Output, 'sample_fft.png'))
	plt.show()
	plt.close()

def spectrogram(spectrogram, time=None, frequencies=None):
	""" Create a simple spectrogram of the sound sample
	"""
	plt.imshow(spectrogram, interpolation=None)

	plt.title('Spectrogram of sound sample')
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')
	plt.xticks(time)
	plt.yticks(frequencies)

	plt.savefig(os.path.join(paths.path2Output, 'sample_spectrogram.png'))
	plt.close()

def gif_projections(tmaps):
	fig, axs = plt.subplots(1, 3)

	axs[0].set_title('4kHz')
	axs[1].set_title('16kHz')
	axs[2].set_title('32kHz')

	axs[0].axis('off')
	axs[1].axis('off')
	axs[2].axis('off')


	ims = []
	for t in range(tmaps.shape[0]):
		im0 = axs[0].imshow(tmaps[t, 0, :, :], cmap='gray', vmin=0, vmax=1)
		im1 = axs[1].imshow(tmaps[t, 1, :, :], cmap='gray', vmin=0, vmax=1)
		im2 = axs[2].imshow(tmaps[t, 2, :, :], cmap='gray', vmin=0, vmax=1)
		#im3 = axs[1, 1].imshow(tmaps[t, 3, :, :], cmap='gray')
		ims.append([im0, im1, im2])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	                                repeat_delay=1000)
	ani.save(os.path.join(paths.path2Output, 'animation.gif'), writer='imagemagick', fps=30)

def figure_1(projection, tmaps, spectro, sample, samplerate, window_ms, overlap):
	window_size = int(window_ms * samplerate * 0.001)
	overlap_size = overlap * 0.01* window_size



	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(3, 4)

	f_spe = fig.add_subplot(gs[0:2, :2])
	f_spe.set_title('Spectrogram')
	f_spe.specgram(sample, Fs=samplerate, NFFT=window_size, noverlap=overlap_size)

	f_wav = fig.add_subplot(gs[-1, :])
	f_wav.set_title('Waveplot')

	display.waveplot(y=sample, sr=samplerate)

	f_wav.set_xlabel('Time (sec)')
	f_wav.set_ylabel('Amplitude')
	f_wav.set_xlim(1, 1.2)

	f_pro1 = fig.add_subplot(gs[0, -2])
	f_pro1.set_title('4kHz')

	f_pro2 = fig.add_subplot(gs[1, -2])
	f_pro2.set_title('16kHz')

	f_pro3 = fig.add_subplot(gs[0, -1])
	f_pro3.set_title('32kHz')

	ims = []
	for t in tqdm(range(tmaps.shape[0])):
		im0 = f_pro1.imshow(projection[t], cmap='gray', vmin=0, vmax=1)
		im1 = f_pro2.imshow(tmaps[t, 1, :, :], cmap='gray')
		im2 = f_pro3.imshow(tmaps[t, 2, :, :], cmap='gray')
		ims.append([im0, im1, im2])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	                                repeat_delay=1000)

	ani.save(os.path.join(paths.path2Output, 'animation.mp4'), 'ffmpeg_file', fps=30)

def rectangle_mp4(min_4, min_32, all_maps):
	fig = plt.figure()
	plt.scatter(min_4[1], min_4[0], marker='o', c='red')
	plt.scatter(min_32[1], min_32[0], marker='o', c='red')
	plt.plot([min_4[1], min_32[1]], [min_4[0], min_32[0]])

	ims = []
	for m in all_maps:
		im0 = plt.imshow(m, vmin=0, vmax=1)
		ims.append([im0])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	                                repeat_delay=1000)
	ani.save(os.path.join(paths.path2Output, 'animation.mp4'), writer='ffmpeg', fps=30)

def cube_show_slider(cube, axis=2, **kwargs):
	    """
	    Display a 3d ndarray with a slider to move along the third dimension.

	    Extra keyword arguments are passed to imshow
	    """
	    import matplotlib.pyplot as plt
	    from matplotlib.widgets import Slider, Button, RadioButtons

	    # check dim
	    if not cube.ndim == 3:
	        raise ValueError("cube should be an ndarray with ndim == 3")

	    # generate figure
	    fig = plt.figure()
	    ax = plt.subplot(111)
	    fig.subplots_adjust(left=0.25, bottom=0.25)

	    # select first image
	    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
	    im = cube[s].squeeze()

	    # display image
	    l = ax.imshow(im, vmin=0, vmax=1)

	    # define slider
	    axcolor = 'lightgoldenrodyellow'
	    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

	    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
	                    valinit=0, valfmt='%i')

	    def update(val):
	        ind = int(slider.val)
	        s = [slice(ind, ind + 1) if i == axis else slice(None)
	                 for i in range(3)]
	        im = cube[s].squeeze()
	        l.set_data(im, **kwargs)
	        fig.canvas.draw()

	    slider.on_changed(update)

	    plt.show()



