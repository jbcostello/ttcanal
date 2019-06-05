import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import scipy.ndimage.filters as filters
from scipy.optimize import curve_fit

class ttc(object):
    '''
    Class to contain all data analysis of ttc (TeraHertz to Thermal Converter) data. 
    The TTC uses a Heimann sensor IR camera, and this code allows analysis of the output 
    of this camera. Although the camera is capable of several file outputs, this 
    code is based on the .txt output
    
    '''

    def __init__(self, inpt, leakfix = False):
        # Turns raw data file or array-like object into
        #   an array of unsized frames, an array of 64x80 frames, and an attribute for 
        # 	number of frames. Also includes an attribute that notes whether the 
        # 	current leak has been accounted for
        #
        # inpt: name of raw data file from camera or array
        # 	function can tell difference between the two
        #
        # Has attributes:
        # self.dat: raw data in np array, each element is a 5120 long frame
        # self.frames: number of frames in the data file
        # self.shapedat: the raw data shaped into  a list of 64x80 2d np arrays. 
        # 	Each element of the list is a shaped frame of the data 
        #
        # except statements allow code to run without errors if improper data is given
        
        if type(inpt) == str:
            try:
                raw = np.genfromtxt(inpt, skip_footer = 2)
                self.dat = raw[:,1:5121]
                self.frames = len(self.dat)
                # this gets rid of all the extra information in the txt file
                # and records number of frames
                try:    
                    self.shapedat = self.dat.reshape((-1, 64, 80))
                except:
                    print("Data from file cannot be shaped into frames.")
                    return False
            except:
                print("Not a viable filename.")
        # This will shape the pixels into the relevant 64x80 format
        else:
            try:
                raw = np.array(inpt)
                try:
                    self.shapedat = raw.reshape((-1, 64, 80))
                    self.dat = raw.reshape((-1, 5120))
                    self.frames = len(self.dat)
                except:
                    print("This array is not the proper shape.")
                    return False
            except:
                print("This is not an array or an array-like object.")
                return False
        # This code will try to make sense of an array you feed as the input. Useful if you 
        # 	want to mess with data then recast it as a TTC object 
        
        self.leakfix = leakfix


    def getframes(self):
		# An easy way to get frame number. Sort of depreciated, can just use self.frames
        print('Number of frames:',self.frames)
        return self.frames

    def animate(self,title=' ',time=200,Verbose=False):
        # Creates an animation of all frames in a given file
        # Returns im_ani animation
        # 
        # title: title of plot (Str)
        # time: time per frame in microseconds (number)
        # Verbose: Boolean for providing extra information
        # At the moment Verbose just tells you number of frames, 
        # 	but could add more in the future

        # To save an animation, do 
        # 	ani = ttc.animate()
        # 	ani.save('name.mp4')

        dat = self.dat
        shpdat = self.shapedat
        frames = self.frames

        if Verbose:
            ttc.getframes(self)

        fig = plt.figure()
        pixmax = np.amax(dat)
        pixmin = np.amin(dat)
        # This is used to set the color scale 
        ims = []
        for frame in np.arange(frames):
            ims.append((plt.pcolormesh(shpdat[frame],vmin = pixmin, vmax = pixmax),))
            im_ani = animation.ArtistAnimation(fig, ims, interval=time, repeat_delay=1000,
            	blit=True)
            # Plots each frame in a pcolormesh then adds it to the animation

        plt.title(title)
        plt.colorbar()
        plt.show()

        return im_ani

    def sdmap(self,title = ' ',Verbose = False,Plot = True):
        # Creates a pcolormesh of the sd across frames 
        # Returns a 2x2 array of the values used to make the map.
        # 
        # title: Title of the std plot (Str)
        # Verbose: Provides additional info for debugging purposes, defaults false (Boolean)
        # Plot = Controls if produces plots, defaults true (Boolean)

        shpdat = self.shapedat
        frames = self.frames 
        
        if Verbose:
            ttc.getframes(self)

        stddata = np.zeros((64,80))
        # initialize to be filled with the standard deviation data

        for i in np.arange(len(stddata[0,:])):
                for j in np.arange(len(stddata[:,0])):
                    stdcalc = np.array([ ])
                    for framenum in np.arange(frames):
                        shapedat = shpdat[framenum]
                        stdcalc = np.append(stdcalc,shapedat[j,i])
                    stddata[j,i] = np.std(stdcalc)
                    # calculates the std across frames for j,i th pixel, loops over 
                    # 	the whole image

        if Plot:
            plt.pcolormesh(stddata)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return stddata

    def sdavgmap(self,sdrange = [1,2,3],sdstep=1,title = ' ',Verbose = False,Plot = True):
        # Creates a pcolormesh of the sd above average. This is computed by taking the 
        # 	average across frames, then finding the mean and std across pixels. 
        # 	The pixels that are above the mean, in steps determined by sdrange, 
        # 	are colored differently. 
		# Returns a 2x2 array of the values used to make the map.
        # 
        # sdrange: array of values of sd that will be used to detect. Ex: [1,2,3] 
        # 	will find all points above 1sd,2sd,3sd. [0.5,1.0,1.5,2.5] will find all 
        # 	points above 0.5 to 2.5 sd. (numpy array)
        # sdstep: each step for a different level in the map. Defaults to one (Float)
        # title: Title of the std plot (Str)
        # Verbose: Provides additional info for debugging purposes (Boolean)
        # Plot = Controls if produces plots, defealts true (Boolean)

        frames = self.frames 
        shpdat = self.shapedat
        # Frames gives you the number of frames in the data file.

        if Verbose:
            ttc.getframes(self)

        detect = np.zeros((64,80))
        avgdat = self.avgmap(Verbose = Verbose, Plot = Verbose)
        if Verbose:
        	print('average data')
        # Returns the data averaged over frames

        mean = np.mean(avgdat)
        std = np.std(avgdat)

        for k in sdrange:
            for i in np.arange(len(detect[0,:])):
                for j in np.arange(len(detect[:,0])):
                    if avgdat[j,i] > mean+k*std: 
                        # This if statement controls what makes a detection, can also add 
                        # 	other if statments to include other things like below standard 
                        # 	deviation etc.
                        detect[j,i] += sdstep
        # So if a given (j,i) pixel is more than k std above the mean, it adds one sd step
        # 	to the detection plot.
        if Plot:
            plt.pcolormesh(detect)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return detect

    def avgmap(self,title = ' ',Verbose = False,Plot = True):
        # Maps the average value across all frames on a pcolormap, and returns the average 
        # 	array
        # 
        # title: title for figure, assumed to be blank (Str)
        # Verbose: gives additional information, assumed false. (Boolean)
        # Plot: whether or not it'll produce a plot
        
        frames = self.frames
        shpdat = self.shapedat

        if Verbose:
            ttc.getframes(self)

        avgdat = np.zeros((64,80))

        for framenum in np.arange(frames):
            shapedat = shpdat[framenum]
            avgdat += shapedat/frames
            # averages together the frames

        if Plot:
            plt.pcolormesh(avgdat)
            plt.title(title)
            plt.colorbar()
            plt.show()

        return avgdat

    def changemap(self,title = ' ',Verbose = False,Plot = True):
        # Shows the net change in values over time from the first frame to the last
        # 
        # title: title for plot, assumed to be blank. (Str)
        # Verbose: gives additional information if True, assumed Flase. (Boolean)
        # Plot: if true produces Plot, assumed true

        shpdat = self.shpdat
        final = np.zeros((64,80))
        
        for i in range(len(shpdat)-1):
            hold = np.subtract(shpdat[i], shpdat[i+1])
            # so this subtracts each frame from the next one? Then checks if it's bigger than
            # 	the previous frame?
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

        # framelist: array that lists which frames to plot/create shaped arrays, 
        # 	defaults to just the first frame (np array)
        # title: title of the plots(Str)
        # Verbose: provides additional information if set to true, defaults to false (Boolean)
        # Plot: creates plots of raw data when set to true, defaults to true (Boolean)

        frames = self.frames
        # Frames gives you the number of frames in the data file.
        shpdat = self.shapedat

        if Verbose:
            ttc.getframes(self)


        shapelist = []

        for framenum in framelist:
            shapedat = shpdat[framenum]
            shapelist.append(shapedat)
            # grabs the frame from framelist and appends it to the shapelist list

            if Plot:
                plt.pcolormesh(shapedat)
                plt.title(title)
                plt.colorbar()
                plt.show()    

        return shapelist

    def diffmap(self, factor, title = ' ',Verbose = False,Plot = True):
        # Shows the net difference between the maximum and minumum values in each pixel,
        #   but ignores the pixels below a certain threshold
        #
        # factor: controls the cutoff threshold, labelled as a factor times the 
        # 	average of the entire dataset
        # title: title of the plots(Str)
        # Verbose: provides additional information if set to true, defaults to false (Boolean)
        # Plot: creates plots of raw data when set to true, defaults to true (Boolean)

        shpdat = self.shapedat
        thing1 = np.zeros((64,80))
        thing2 = np.zeros((64,80)) + 100000
        
        for i in range(len(shpdat)):
            thing1 = np.maximum(thing1, shpdat[i])
            thing2 = np.minimum(thing2, shpdat[i])
        # Ok so this keeps a running total of the max, min value for each pixel

        raw = thing1 - thing2
        raw[raw < factor*np.average(raw)] = 0
        # This give the difference between min and max for each pixel, with a cutoff 
        # 	determined by factor
   
        if Plot:
            plt.pcolormesh(raw, cmap="terrain")
            plt.title(title)
            plt.colorbar()
            plt.show()

        return raw
        
    def areaTry(self, factor, title = ' ',Verbose = False,Plot = True):
        # Similar to diffmap, but smears what would be the final image to make areas 
        #   of similar values more visually obvious
        # 
        # factor: controls the cutoff threshold, labelled as a factor times the 
        # 	average of the entire dataset
        # title: title of the plots(Str)
        # Verbose: provides additional information if set to true, defaults to false (Boolean)
        # Plot: creates plots of raw data when set to true, defaults to true (Boolean)


        shpdat = self.shapedat
        thing1 = np.zeros((64,80))
        thing2 = np.zeros((64,80)) + 100000
        
        for i in range(len(shpdt)):
            thing1 = np.maximum(thing1, shpdat[i])
            thing2 = np.minimum(thing2, shpdat[i])

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
        
    def calib_max(self):   
        # A function to retrieve calibration data from Toothpick Jig testing of pixel size
        #
        # returns the mean of distances between high points in each frame, 
        #   the standard deviation, and each frame with the relevant points marked
        maxfilter, maxima = np.copy(self.shapedat), np.copy(self.shapedat)
        distances, size = {}, 15
        maxframes = []
        
        for i in range(len(self.shapedat)):
            maxfilter[i] = filters.maximum_filter(self.shapedat[i], size)
            maxima[i] = (self.shapedat[i] == maxfilter[i])
            scale = np.amax(self.shapedat[i]) - np.mean(self.shapedat[i])
            if scale > 1.95:
                factor = 0
                maxima[i][self.shapedat[i] < np.mean(self.shapedat) + scale*factor] = 0
                while len(np.where(maxima[i] == True)[0]) > 2:
                    factor += 0.01
                    maxima[i][self.shapedat[i] < np.mean(self.shapedat) + scale*factor] = 0
            
                dis = (np.where(maxima[i] == True))
                try:
                    distances[i] = ((dis[0][0]-dis[0][1])**2 + (dis[1][0]-dis[1][1])**2)**(0.5)
                    maxframes.append(maxima[i])
                except:
                    pass
        
        mean = np.mean(list(distances.values()))
        std = np.std(list(distances.values()))
        
        return mean, std, maxframes

    def fix_leak(self, background):
        # Accepts a recording of background data to eliminate current leak in data 
        # 	taken from camera
        
        mask = background.avgmap(Plot = False) - np.amin(background.avgmap(Plot = False))
        
        if self.leakfix == False:
            new = self.shapedat - mask
            fix = ttc(new, leakfix = True)
            return fix
        else:
            return self
    
    def row_ani(self, line = "row 0", title=' ',time=200,terms=15):
        # Creates an over-time animation of the line plot of a given row or column
        # Defaults to the bottom row
        #
        # To save an animation, do 
        # ani, plot_data = ttc.row_ani(line = "row/col #")
        # ani.save('name.mp4')
        
        guide = {"row":80, "col":64}
        
        inpt = line.split()
        if (((len(inpt) == 2) and (inpt[0] in guide)) and (type(inpt[1]==int))):
            coord = inpt[0]
            num = abs(int(inpt[1]))
            if (num < guide[coord]):
                pass
            else:
                print("This is not a valid entry.")
                return False 
        else:
            print("This is not a valid entry.")
            return False
        
        def fourier(x, *a):
            ret = 0
            for deg in range(len(a)):
                ret += a[deg] * np.cos(((deg) * np.pi * x)/ guide[coord])
            return ret
        
        if guide[coord] == 80:
            data = [self.shapedat[:, num],[]]
        elif guide[coord] == 64:
            data = [self.shapedat[:, :, num],[]]

        hold = [curve_fit(fourier, np.arange(guide[coord]), data[0][i], [1.0] * terms)[0] for i in range(guide[coord])]
        data[1] = [[np.sum([(hold[k][i]*np.cos((i*np.pi*j)/guide[coord])) for i in range(terms)]) for j in range(guide[coord])] for k in range(len(hold))]
            
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylim(np.amin(data[0])-1, np.amax(data[0])+1)
        
        ims = []
        for i in range(guide[coord]):
            im1, = ax1.plot(range(guide[coord]), data[0][i], color="black")
            im2, = ax1.plot(range(guide[coord]), data[1][i], color="red")
            ims.append([im1, im2])
        
        im_ani = animation.ArtistAnimation(fig, ims, interval=time, repeat_delay=1000,blit=True)

        plt.show()
        
        return im_ani

    def plot_smudge(self, num):
        # Averages a number of adjacent frames to try to increase SNR
        # Number of frames will match input ttc length by adding last frame as many times as necessary
        #
        # num: number of frames to be averaged
        
        if num > self.frames:
            return False
        
        smudged = [(np.sum(self.shapedat[i: i+num], axis=0)/num) for i in range(self.frames-(num-1))]
        
        print(type(smudged))
        for i in range(num-1):
            smudged.append(smudged[-1])
        
        print(type(smudged))
        return ttc(smudged)
        
        

######################################################################################### 
######################################################################################### 
######################################################################################### 










