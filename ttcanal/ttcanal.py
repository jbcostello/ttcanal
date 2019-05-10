import numpy as np
#import hsganalysis.ipg as pg
import matplotlib.pylab as plt
#import glob
#import os
#import json
#import hsganalysis as hsg
#from hsganalysis import newhsganalysis
#import scipy as sp 
import matplotlib.animation as animation
from scipy import ndimage

class ttc(object):
    '''
    Class to contain all data analysis of ttc (TeraHertz to Thermal Converter) data. The TTC uses a Heimann sensor IR camera, 
    and this code allows analysis of the output of this camera. Although the camera is capable of several file outputs, this code is based on the .txt output
    
    '''

    def __init__(self,file):
        # Turns raw data file into array of unsized frames, and creates an attribute for number of frames in a given frame.
        # 
        # file: name of raw data file from camera (Str)
        # 
        # Has attributes:
        # self.dat: raw data in np array, each element is a 5120 long frame
        # self.frames: number of frames in the data file
        # self.shapedat: the raw data shaped into  a list of 64x80 2d np arrays. Each element of the list is a shaped frame of the data 

        rawdat = np.genfromtxt(file, skip_footer = 2)
        self.dat = rawdat[:,1:5121]
        # This produces an array, each element of which is 1520 pixel values forming a frame. Each element can be resized to 64x80 for plotting purposes 

        self.frames = len(self.dat[:,0])
        # This gets the number of frames, useful in later calculations

        self.shapedat = []
        for frame in np.arange(self.frames):
            self.shapedat.append(np.reshape(self.dat[frame,:],(64,80)))
        # creates a list of 64x80 np arrays, each one of which corresponds to a single frame of data



    def getframes(self):

        print('Number of frames:',self.frames)

        return self.frames

    def animate(self,title=' ',time=200,Verbose=False):
        # Creates an animation of all frames in a given file
        # Returns im_ani animation
        # 
        # file: filename of raw data from camera (Str)
        # title: title of plot (Str)
        # time: time per frame in microseconds (number)
        # Verbose: Boolean for providing extra information
        # At the moment Verbose just tells you number of frames, but could add more in the future

        # To save an animation, do 
        # ani = ttc.animate()
        # ani.save('name.mp4')

        dat = self.dat
        shpdat = self.shapedat
        frames = self.frames

        if Verbose:
            ttc.getframes(self)

        fig = plt.figure()

        pixmax = np.amax(dat)
        pixmin = np.amin(dat)
        ims = []
        for frame in np.arange(frames):
            ims.append((plt.pcolormesh(shpdat[frame],vmin = pixmin, vmax = pixmax),))
            # ims.append((plt.pcolormesh(np.reshape(dat[frame,:],(64,80)),vmin = pixmin, vmax = pixmax),))
            im_ani = animation.ArtistAnimation(fig, ims, interval=time, repeat_delay=1000,blit=True)
            # Changed to use shapedat instead of raw dat. Old way works, but this is more straightforward I feel.

        plt.title(title)
        plt.colorbar()
        plt.show()

        return im_ani

    def sdmap(self,title = ' ',Verbose = False,Plot = True):
        # Creates a pcolormesh of the sd across frames 
        # Returns a 2x2 array of the values used to make the map.
        # 
        # file: filename of raw data from camera (Str)
        # title: Title of the std plot (Str)
        # Verbose: Provides additional info for debugging purposes, defaults false (Boolean)
        # Plot = Controls if produces plots, defaults true (Boolean)

        dat = self.dat
        shpdat = self.shapedat
        frames = self.frames 
        
        if Verbose:
            ttc.getframes(self)

        stddata = np.zeros((64,80))

        for i in np.arange(len(stddata[0,:])):
                for j in np.arange(len(stddata[:,0])):
                    stdcalc = np.array([ ])
                    for framenum in np.arange(frames):
                        shapedat = shpdat[framenum]
                        stdcalc = np.append(stdcalc,shapedat[j,i])
                    stddata[j,i] = np.std(stdcalc)

        if Plot:
            plt.pcolormesh(stddata)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return stddata

    def sdavgmap(self,sdrange = [1,2,3],sdstep=1,title = ' ',Verbose = False,Plot = True):
        # Creates a pcolormesh of the sd above average. This is computed by taking the average across frames, then finding the mean and std across pixels. The pixels that are above the mean, in steps determined by sdrange, are colored differently
        # Returns a 2x2 array of the values used to make the map.
        # 
        # file: filename of raw data from camera (Str)
        # sdrange: array of values of sd that will be used to detect. Ex: [1,2,3] will find all points above 1sd,2sd,3sd. [0.5,1.0,1.5,2.5] will find all points above 0.5 to 2.5 sd. (numpy array)
        # sdstep: each step for a different level in the map. Defaults to one (Float)
        # title: Title of the std plot (Str)
        # Verbose: Provides additional info for debugging purposes (Boolean)
        # Plot = Controls if produces plots, defealts true (Boolean)

        dat = self.dat
        frames = self.frames 
        shpdat = self.shapedat
        # Frames gives you the number of frames in the data file.

        if Verbose:
            ttc.getframes(self)

        detect = np.zeros((64,80))
        avgdat = np.zeros((64,80))

        for framenum in np.arange(frames):
            shapedat = shpdat[framenum]
            avgdat += shapedat/frames

        mean = np.mean(avgdat)
        std = np.std(avgdat)

        for k in sdrange:
            for i in np.arange(len(detect[0,:])):
                for j in np.arange(len(detect[:,0])):
                    if avgdat[j,i] > mean+k*std: 
                        # This if statement controls what makes a detection, can also add other if statments to include other things like below standard deviation etc.
                        detect[j,i] += sdstep
        if Plot:
            plt.pcolormesh(detect)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return detect

    def avgmap(self,title = ' ',Verbose = False,Plot = True):
        # Maps the average value across all frames on a pcolormap, and returns the average array
        # 
        # file: filename for raw data from the camera (Str)
        # title: title for figure, assumed to be blank (Str)
        # Verbose: gives additional information, assumed false. (Boolean)
        # Plot: whether or not it'll produce a plot
        
        dat = self.dat
        frames = self.frames
        # Frames gives you the number of frames in the data file.
        shpdat = self.shapedat

        if Verbose:
            ttc.getframes(self)

        avgdat = np.zeros((64,80))

        for framenum in np.arange(frames):
            # shapedat = np.reshape(dat[framenum,:],(64,80))
            # reshapes the data to the appropriate size for getting images.
            shapedat = shpdat[framenum]
            avgdat += shapedat/frames

        if Plot:
            plt.pcolormesh(avgdat)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return avgdat

    def changemap(self,title = ' ',Verbose = False,Plot = True):
        stuff = self.shapedat
        final = np.zeros((64,80))
        
        for i in range(len(stuff)-1):
            hold = np.subtract(stuff[i], stuff[i+1])
            hold[hold > 0] = 1
            hold[hold < 0] = -1
            final = np.add(final, hold)
    
        if Plot:
            plt.pcolormesh(final, cmap="terrain")
            plt.title(title)
            plt.colorbar()
            plt.show()

        return final


    def rawmap(self,framelist = np.array([0]),title = ' ',Verbose = False,Plot = True):
        # Creates arrays of raw frames and plots them
        # Returns a list of raw 2x2 numpy arrays correctly shaped

        # file: filename for raw data from the camera (Str)
        # framelist: array that lists which frames to plot/create shaped arrays, defaults to just the first frame (np array)
        # title: title of the plots(Str)
        # Verbose: provides additional information if set to true, defaults to false (Boolean)
        # Plot: creates plots of raw data when set to true, defaults to true (Boolean)

        dat = self.dat
        frames = self.frames
        # Frames gives you the number of frames in the data file.
        shpdat = self.shapedat

        if Verbose:
            ttc.getframes(self)


        shapelist = []

        for framenum in framelist:
            shapedat = shpdat[framenum]
            shapelist.append(shapedat)

            if Plot:
                plt.pcolormesh(shapedat)
                plt.title(title)
                plt.colorbar()
                plt.show()    

        return shapelist

    def diffmap(self, factor, title = ' ',Verbose = False,Plot = True):
        stuff = self.shapedat
        thing1 = np.zeros((64,80))
        thing2 = np.zeros((64,80)) + 100000
        
        for i in range(len(stuff)):
            thing1 = np.maximum(thing1, stuff[i])
            thing2 = np.minimum(thing2, stuff[i])

        raw = thing1 - thing2
        raw[raw < factor*np.average(raw)] = 0
    
        if Plot:
            plt.pcolormesh(raw, cmap="terrain")
            plt.title(title)
            plt.colorbar()
            plt.show()

        return raw
        
    def areaTry(self, factor, title = ' ',Verbose = False,Plot = True):
        stuff = self.shapedat
        thing1 = np.zeros((64,80))
        thing2 = np.zeros((64,80)) + 100000
        
        for i in range(len(stuff)):
            thing1 = np.maximum(thing1, stuff[i])
            thing2 = np.minimum(thing2, stuff[i])

        raw = thing1 - thing2
        floor = np.average(raw)
        raw[raw < factor*floor] = 0
        raw[raw >= factor*floor] = 1
        final = np.zeros((64,80))
        
        for i in range(2):
            for j in range(2):
                final += np.roll(raw, (-1)**i, axis = j)
                final += np.roll(np.roll(raw, (-1)**i, axis = 0), (-1)**j, axis = 1)
                
        final = (final + raw) / 9
    
        if Plot:
            plt.pcolormesh(final[2:-2, 2:-2], cmap="terrain")
            plt.title(title)
            plt.colorbar()
            plt.show()

        return raw
        
######################################################################################### 
######################################################################################### 
######################################################################################### 










