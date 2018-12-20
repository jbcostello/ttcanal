import numpy as np
import interactivePG as pg
import matplotlib.pylab as plt
import glob
import os
import json
import hsganalysis as hsg
from hsganalysis import newhsganalysis
import scipy as sp 
import matplotlib.animation as animation




def getttcdata(file):
	# Turns raw data file into array of unsized frames
	# 
	# file: name of raw data file from camera (Str)

	rawdat = np.genfromtxt(file, skip_footer = 2)
	dat = rawdat[:,1:5121]

	return dat 

def getframes(file,Verbose = False):
	# Returns number of frames in a given file
	# 
	# file: filename for raw data from the camera (Str)

	dat = getttcdata(file)
	frames = len(dat[:,0])

	if Verbose: 
		print('Number of Frames')
		print(frames)
	
	return frames

def animate(file,title = ' ',time = 200,Verbose = False):
	# Creates an animation of all frames in a given file
	# Returns im_ani animation
	# 
	# file: filename of raw data from camera (Str)
	# title: title of plot (Str)
	# time: time per frame in microseconds (number)
	# Verbose: Boolean for providing extra information
	# At the moment Verbose just tells you number of frames, but could add more in the future
	



	dat = getttcdata(file)
	frames = getframes(file,Verbose)

	fig = plt.figure()

	pixmax = np.amax(dat)
	pixmin = np.amin(dat)
	ims = []
	for frame in np.arange(frames):
		ims.append((plt.pcolormesh(np.reshape(dat[frame,:],(64,80)),vmin = pixmin, vmax = pixmax),))
		im_ani = animation.ArtistAnimation(fig, ims, interval=time, repeat_delay=1000,blit=True)

	plt.title(title)
	plt.colorbar()
	plt.show()

	return im_ani

def sdmap(file,sdrange,sdstep=1,title1 = ' ',title2 = ' ',Verbose = False,Plot = True):
	# Creates 2 pcolormeshs. One of the standard deviations and one of standard deviations above the mean. Takes average among frames and shows standard dev. 
	# Returns 2 2x2 arrays of the values used to make the map. 1 is map of std values, 2 is map of std values above the mean in steps provided by sdrange
	# 
	# file: filename of raw data from camera (Str)
	# sdrange: array of values of sd that will be used to detect. Ex: [1,2,3] will find all points above 1sd,2sd,3sd. [0.5,1.0,1.5,2.5] will find all points above 0.5 to 2.5 sd. (numpy array)
	# sdstep: each step for a different level in the map. Defaults to one (Float)
	# title1: Title of the std plot (Str)
	# title2: Title of the std above average plot (Str)
	# Verbose: Provides additional info for debugging purposes (Boolean)
	# Plot = Controls if produces plots, defealts true

	dat = getttcdata(file)
	frames = getframes(file,Verbose) 
	# Frames gives you the number of frames in the data file.

	avgdat = np.zeros((64,80))
	detect = np.zeros((64,80))

	for framenum in np.arange(frames):
		shapedat = np.reshape(dat[framenum,:],(64,80))
		# reshapes the data to the appropriate size for getting images.
		avgdat += shapedat/frames

	stddata = np.zeros((64,80))

	for i in np.arange(len(avgdat[0,:])):
			for j in np.arange(len(avgdat[:,0])):
				stdcalc = np.array([ ])
				for framenum in np.arange(frames):
					shapedat = np.reshape(dat[framenum,:],(64,80))
					stdcalc = np.append(stdcalc,shapedat[j,i])
				stddata[j,i] = np.std(stdcalc)

	if Plot:
		plt.pcolormesh(stddata)
		plt.title(title1)
		plt.colorbar()
		plt.show()



	for k in sdrange:
				if avgdat[j,i] > mean+k*std: 
					# This if statement controls what makes a detection, can also add other if statments to include other things like below standard deviation etc.
					detect[j,i] += sdstep
	if Plot:
		plt.pcolormesh(detect)
		plt.title(title2)
		plt.colorbar()
		plt.show()

	return stddata, detect

def avgmap(file,title = ' ',Verbose = False,Plot = True):
	# Maps the average value across all frames on a pcolormap, and returns the average array
	# 
	# file: filename for raw data from the camera (Str)
	# title: title for figure, assumed to be blank (Str)
	# Verbose: gives additional information, assumed false. (Boolean)
	# Plot: whether or not it'll produce a plot
	
	dat = getttcdata(file)
	frames = getframes(file,Verbose)
	# Frames gives you the number of frames in the data file.
	avgdat = np.zeros((64,80))

	for framenum in np.arange(frames):
		shapedat = np.reshape(dat[framenum,:],(64,80))
		# reshapes the data to the appropriate size for getting images.
		avgdat += shapedat/frames

	if Plot:
		plt.pcolormesh(avgdat)
		plt.title(title)
		plt.colorbar()
		plt.show()

	return avgdat

def rawmap(file,framelist = np.array([0]),title = ' ',Verbose = False,Plot = True):
	# Creates arrays of raw frames and plots them
	# Returns a list of raw 2x2 numpy arrays correctly shaped

	# file: filename for raw data from the camera (Str)
	# framelist: array that lists which frames to plot/create shaped arrays, defaults to just the first frame (np array)
	# title: title of the plots(Str)
	# Verbose: provides additional information if set to true, defaults to false (Boolean)
	# Plot: creates plots of raw data when set to true, defaults to true (Boolean)

	dat = getttcdata(file)
	frames = getframes(file,Verbose)
	# Frames gives you the number of frames in the data file.

	shapelist = []

	for framenum in framelist:
		shapedat = np.reshape(dat[framenum,:],(64,80))
		shapelist.append(shapedat)

		if Plot:
			plt.pcolormesh(shapedat)
			plt.title(title)
			plt.colorbar()
			plt.show()	

	return shapelist



fname = 'grid1hz.txt'
frames = getframes(fname)

# ani = animate(fname,time = 100,title = 'Looking for a Grid')
# ani.save('grid.mp4')



sdmap(fname,sdrange = np.arange(0.5,4,0.5),sdstep = 0.5,title1='$\sigma$ Across Frames Looking for Grid',title2='$\sigma$ Above Mean Looking for Grid')
# avgmap(fname)

# frlist = np.arange(0,frames,20)
# rawmap(fname,frlist)







