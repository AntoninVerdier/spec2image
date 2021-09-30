import os 
import librosa
import warnings
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import plot as pl 
import settings as sett
import processing as proc

from functions.Sample import Sound
from matplotlib.widgets import Slider

warnings.filterwarnings("ignore")

paths = sett.paths()
params = sett.parameters()
args = sett.arguments().args

for file in os.listdir(paths.path2Sample):
	print(file)
	#Extract data
	sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'PT_7300Hz_500ms_70dB.wav'),
									  sr=None, mono=True, offset=0.0, duration=None)

	tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))
	tonotopic_maps = proc.downscale_tmaps(tonotopic_maps, block_size=(4, 4))

	# Remove weighted map at the end and white noise at the beginning

	tonotopic_maps = tonotopic_maps[1:-1, :, :]

	# Normalization of tonotopic maps and inversion (bright spots should be of interest)
	for i, tmap in enumerate(tonotopic_maps):
		tonotopic_maps[i] = 1 - (tmap - np.min(tmap))/(np.max(tmap) - np.min(tmap))

	# If sample is in stereo take only one track
	if sample.ndim > 1:
		sample = np.transpose(sample[:-len(sample)/2, 0])

	# Compute spectrogram
	specgram, frequencies, times = proc.spectro(sample, samplerate, plot=True)

	# Compute rectangle stimulation if needed
	if args.rectangle is not None:
		rect_stim, weighted_tmap, min_4, min_32 = proc.rectangle_stim(tonotopic_maps[0], tonotopic_maps[2], args.rectangle)
		#frequencies = np.linspace(4000, 32000, num=801)
		magnitudes = proc.rectangle_windowing(specgram, frequencies, n_rectangle=args.rectangle)


		all_maps = []
		for magnitude in magnitudes:
			buffer_map = np.copy(weighted_tmap)
			for rect, mag in zip(rect_stim, magnitude):
				buffer_map[rect[:, 0], rect[:, 1]] = mag
			all_maps.append(buffer_map)
		all_maps = np.array(all_maps)
		all_maps = all_maps[:400]
		
		if args.plot:
			pl.rectangle_mp4(min_4, min_32, all_maps)
		single_map = True

	else:
		magnitudes = proc.gaussian_windowing_updated(specgram, frequencies)
		all_maps = np.array([tonotopic_maps * mag[:, np.newaxis, np.newaxis] for mag in magnitudes])
		single_map = False


	projections = proc.implant_projection(all_maps, single_map=single_map)
	np.save(os.path.join(paths.path2Output, file[:-4]), projections)
	print(projections.shape)

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
	    l = ax.imshow(im, **kwargs)

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

	cube_show_slider(np.swapaxes(projections[1:,:,:], 0, 2))

# Do a different processinf for AM sound
# Revoir le gaussian windowing qui est chelou. Fréquence pas stable. 