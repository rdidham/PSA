# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:48:10 2021

@author: Richard Didham
GUI to be used with performance sail analysis computer vision program. Gathers
and settings then launches PSA script

Version notes:
    v1.0, combines frames 6 and 5, then includes entry box to allow users to 
        specify the number of draft stripes in on their sails
    v1.1, grid frame 5 instead of pack. cleaned up some other stuff
"""
import tkinter as tk
import tkinter.ttk as ttk
import PSA_lib as lib
import os, glob
from tkinter import filedialog
import csv
import numpy as np
import cv2

class GUI:
    def __init__(self):
        self.path = os.path
        #initialize window object
        self.window = tk.Tk()
        self.window.title("Performance Sail Anaysis")
        self.runFlag = tk.BooleanVar(value=False)
        self.calLoaded = False
        self.imagePathLoaded = False
        self.minHueDef = 45
        self.maxHueDef = 90
        self.minSatDef = 25
        self.maxSatDef = 255
        self.minValue = 0
        self.maxValue = 255
        
        #quick couple lines to add menu at top
        menuBar = tk.Menu(self.window)
        self.window.config(menu=menuBar)
        helpMenu = tk.Menu(menuBar, tearoff=0)
        helpMenu.config(borderwidth=2)
        helpMenu.add_command(label='Instructions', command=self.instructions)
        helpMenu.add_command(label='About', command=self.about)
        menuBar.add_cascade(label='Help Menu', menu=helpMenu)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # First Frame: button to launch program that creates camera calibration
        # files
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.frame1 = tk.Frame(master=self.window, padx=5, pady=0)
        self.btnCreateCamCal = tk.Button(master=self.frame1,
                                     text='Create Camera Calibration File',
                                     command=self.createCamCal)
        self.msgFrame1 = tk.Label(master=self.frame1, 
                                  text='Note: only select if a camera calibration'
                                  ' file\n has not previously been created')
        self.lineFrame1 = tk.Label(master=self.frame1,
                                   text='___________________________________________________________')
        
        self.btnCreateCamCal.pack()
        self.msgFrame1.pack()
        self.lineFrame1.pack()
        self.frame1.pack()
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Second Frame: button that allows users to select a calibration file
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.frame2 = tk.Frame(master=self.window, padx=5, pady=0)
        self.camCalLoaded = False #initialize boolean indicating no file loaded
        self.btnLoadCamCal = tk.Button(master=self.window, 
                                       text='Load Camera Calibration File',
                                       command=self.loadCamCal)
        self.msgFrame2 = tk.Label(master=self.frame2,
                                  text='Calibration file not loaded')
        self.lineFrame2 = tk.Label(master=self.frame2,
                                   text='___________________________________________________________')
        self.btnLoadCamCal.pack()
        self.msgFrame2.pack()
        self.lineFrame2.pack()
        self.frame2.pack()
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Third Frame: Set Threshold Parameters
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #boolean variable whether default threshold values are being used
        self.custParam = tk.BooleanVar(value=False) 
        self.frame3 = tk.Frame(master=self.window, padx=5, pady=0)
        self.frame3a = tk.Frame(master=self.frame3)
        self.frame3b = tk.Frame(master=self.frame3)
        self.msgFrame3 = tk.Label(master=self.frame3,
                                  text='Image Threshold Parameters')
        #radio button for default threshold parameters for green stripes
        self.defRdBtn = tk.Radiobutton(master=self.frame3a, text='Default',
                                       variable=self.custParam, value=False,
                                       command=self.setDefaultParams)
        #radio button to allow threshold parameters to be changed manually
        self.custRdBtn = tk.Radiobutton(master=self.frame3a, text='Custom',
                                        variable=self.custParam, value=True,
                                        command=self.setCustParams)
        #button that launches interactive method to set threshold parameters
        #a model image to be threshold-ed is loaded and the user can use track-
        #bars to determine desired parameters
        self.interactiveBtn = tk.Button(master=self.frame3a, text='Interactive',
                                        command=self.interactiveThreshold)
        
        #create entry boxes that will contain image threshold parameters
        self.minHueEnt = tk.Entry(master=self.frame3b, width=3)
        self.maxHueEnt = tk.Entry(master=self.frame3b, width=3)
        self.minSatEnt = tk.Entry(master=self.frame3b, width=3)
        self.maxSatEnt = tk.Entry(master=self.frame3b, width=3)
        #insert default parameters in boxes
        self.minHueEnt.insert(0, str(self.minHueDef))
        self.maxHueEnt.insert(0, str(self.maxHueDef))
        self.minSatEnt.insert(0, str(self.minSatDef))
        self.maxSatEnt.insert(0, str(self.maxSatDef))
        #disable boxes initially in default mode. Once the user clicks the 
        #"custom" or "interactive" buttons, these boxes will switch to normal 
        self.minHueEnt.config(state=tk.DISABLED)
        self.maxHueEnt.config(state=tk.DISABLED)
        self.minSatEnt.config(state=tk.DISABLED)
        self.maxSatEnt.config(state=tk.DISABLED)
        
        #labels to be used to organize the entry boxes
        self.hueLbl = tk.Label(master=self.frame3b, text='Hue')
        self.satLbl = tk.Label(master=self.frame3b, text='Saturation')
        self.lowerLbl = tk.Label(master=self.frame3b, text='Lower Bound')
        self.upperLbl = tk.Label(master=self.frame3b, text='Upper Bound')
        
        self.msgFrame3ab = tk.Label(master=self.frame3,
                                    text='- - - -')
        self.msgFrame3ab2 = tk.Label(master=self.frame3,
                                    text='the following parameters will be used:')
        
        self.lineFrame3 = tk.Label(master=self.frame3,
                                   text='___________________________________________________________')
        
        #pack (or grid) everything together
        self.msgFrame3.pack()
        self.defRdBtn.pack(side='left')
        self.custRdBtn.pack(side='left')
        self.interactiveBtn.pack(side='left')
        self.frame3a.pack()
        self.msgFrame3ab.pack()
        self.msgFrame3ab2.pack()
        self.hueLbl.grid(row=0, column=1, padx=3, pady=3)
        self.satLbl.grid(row=0, column=2, padx=3, pady=3)
        self.lowerLbl.grid(row=1, column=0, padx=3, pady=3)
        self.minHueEnt.grid(row=1, column=1, padx=3, pady=3)
        self.minSatEnt.grid(row=1, column=2, padx=3, pady=3)
        self.upperLbl.grid(row=2, column=0, padx=3, pady=3)
        self.maxHueEnt.grid(row=2, column=1, padx=3, pady=3)
        self.maxSatEnt.grid(row=2, column=2, padx=3, pady=3)
        
        
        self.frame3b.pack()
        self.lineFrame3.pack()
        self.frame3.pack()
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fourth Frame: Choose Directory of Images
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.frame4 = tk.Frame(master=self.window, padx=5, pady=0)
        self.imagesAreLoaded = False
        self.btnLoadImages = tk.Button(master=self.frame4,
                                text='Choose Directory of Images to Process',
                                command=self.loadImages)
        self.msgFrame4 = tk.Label(master=self.frame4, 
                                  text='No images have been loaded')
        self.lineFrame4 = tk.Label(master=self.frame4,
                                   text='___________________________________________________________')
        
        self.btnLoadImages.pack()
        self.msgFrame4.pack()
        self.lineFrame4.pack()
        self.frame4.pack()
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fith Frame: Ask a couple questions related to the program
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.reduceImageSize = tk.BooleanVar(value=False)
        self.frame5 = tk.Frame(master=self.window, padx=5, pady=0)
        self.frame5grid = tk.Frame(master=self.frame5)
        self.frame5BtnImageReduct = tk.Frame(master=self.frame5grid)
        self.frame5BtnOverlayImages = tk.Frame(master=self.frame5grid)
        
        self.msgTopFrame5 = tk.Label(master=self.frame5,
                            text='Other User Settings')
        self.msgImageReduct = tk.Label(master=self.frame5grid,
                            text='Reduce image resolution? \n(speeds up run '
                            'time but reduces accuracy)')
        self.btnNoImageReduct = tk.Radiobutton(master=self.frame5BtnImageReduct,
                                    text='no', variable=self.reduceImageSize,
                                    value=False)
        self.btnYesImageReduct = tk.Radiobutton(master=self.frame5BtnImageReduct,
                                    text='yes', variable=self.reduceImageSize,
                                    value=True)
        
        #radio button yes/no whether user wants to output overlaid images 
        self.overlayImages = tk.BooleanVar(value=False)        
        self.msgOverlayImages = tk.Label(master=self.frame5grid,
                            text='Overlay results on sail images?')
        self.btnDontOverlayImages = tk.Radiobutton(
                                    master=self.frame5BtnOverlayImages,
                                    text='no', variable=self.overlayImages,
                                    value=False)
        self.btnOverlayImages = tk.Radiobutton(master=self.frame5BtnOverlayImages,
                                    text='yes', variable=self.overlayImages,
                                    value=True)
        
        #entry to set number of draft stripes (default=3)
        self.msgNumStripes = tk.Label(master=self.frame5grid,
                                      text='Number of draft stripes on sail?')
        self.numStripesEnt = tk.Entry(master=self.frame5grid, width=2)
        self.numStripesEnt.insert(0, str(3))
        
        self.lineFrame5 = tk.Label(master=self.frame5,
                                   text='___________________________________________________________')
        
        self.msgTopFrame5.pack()
        gridpadx = 2
        gridpady = 3
        self.msgImageReduct.grid(row=0, column=0, padx=gridpadx, pady=gridpady)
        self.btnNoImageReduct.pack(side='left')
        self.btnYesImageReduct.pack(side='left')
        self.frame5BtnImageReduct.grid(row=0, column=1, padx=gridpadx, pady=gridpady)
        
        self.msgOverlayImages.grid(row=1, column=0)
        self.btnDontOverlayImages.pack(side='left')
        self.btnOverlayImages.pack(side='left')
        self.frame5BtnOverlayImages.grid(row=1, column=1, padx=gridpadx, pady=gridpady)
        
        self.msgNumStripes.grid(row=2, column=0, padx=gridpadx, pady=gridpady)
        self.numStripesEnt.grid(row=2, column=1, padx=gridpadx, pady=gridpady)
        
        self.frame5grid.pack()
        self.lineFrame5.pack()
        self.frame5.pack()
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Seventh Frame: Run Program!
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.frame6 = tk.Frame(master=self.window, padx=5, pady=5)
        self.frame6btn = tk.Frame(master=self.frame6, pady=2)
        self.frame6prog = tk.Frame(master=self.frame6, pady=2)
        self.btnRun = tk.Button(master=self.frame6btn,
                                text='Start Sail Shape Analysis',
                                command=self.run)
        self.runProgress = ttk.Progressbar(master=self.frame6prog,
                                           orient=tk.HORIZONTAL, 
                                           length=100,
                                           mode='determinate')
        
        self.btnRun.pack()
        self.runProgress.pack()
        
        self.frame6btn.pack()
        self.frame6prog.pack()
        self.frame6.pack()
        
        self.window.mainloop()
   
    #~~~~~~~~~~~~~~~~~~~ Class Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #two functions for help menu commands. should probably add more insturctions
    def instructions(self):
        tk.messagebox.showinfo('Instructions', '"An informative message."'
                               '\n \n Please contact Richard Didham for more '
                               'information')
        
    def about(self):
        tk.messagebox.showinfo('About', 'Program developed by Richard Didham '
                               'to automatically calculate sail shape ' 
                               'parameters from draft stripes.\n'  
                               'Version: 1.1.0     Release date: 6-6-21')
    
    #launches script to create camera calibration file from checkerboard images    
    def createCamCal(self):
        #first, ask for directory containing checkerboard images to be used for
        #calibration
        dialogTitle = 'Select a file folder containing calibration images'
        selectedFolder = filedialog.askdirectory(title=dialogTitle)
        
        #one way to find all the images files in given directory
        # images = []
        # for file in os.listdir(selectedFolder):
        #     if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
        #         images.append(file)
        
        #second way to find all the image files in given directory
        os.chdir(selectedFolder)
        images = (glob.glob("*.jpg") + glob.glob("*.png") + 
                  glob.glob("*.JPG") + glob.glob("*.PNG"))
        
        #return if no images are found in the directory 
        if images == []:
            tk.messagebox.showerror('No Image Error', 'No image files were '
                                    'found. Please check the selected folder'
                                    ' again.')
            return
        
        #message to display number of images that will be processed
        processingMsg = (str(len(images)) + ' image files were found. Camera' 
                         ' calibration currently\n in progress. This may take a' 
                         ' minute or two depending\n on the size and quantity'
                         ' of images used.')
        self.msgFrame1["text"] = processingMsg #update GUI label
        self.msgFrame1.update_idletasks()
        
        #try to process images and return error if unable to
        try:
            numImagesUsed = lib.calibrate(images) #call calibration function
        except:
            tk.messagebox.showerror('Image Processing Error', 'The calibration '
                                    'program encountered an error while processing'
                                    ' images. Please check the images in the '
                                    'selected folder all have the same resolution'
                                    ' and are suitable for calibration')
            return
        
        #report number of images used during camera calibration
        self.msgFrame1["text"] = (str(numImagesUsed) + ' images were successfully used'
                                  ' to create the camera calibration file')
        self.msgFrame1.update_idletasks()
        
        completionMsg = (str(numImagesUsed) + ' images were successfully used in '
                         'camera calibration. Note, if this is significantly'
                         ' fewer\n than expected, you may want to try re-taking the '
                         'checkerboard images used for camera calibration.')
        tk.messagebox.showinfo(message=completionMsg)
    
    #function to load the camera calibration file to be used
    def loadCamCal(self):
        #ask user for path of calibration file
        titleStr = 'Select Calibration File'
        fileTypeTuple = (('calibration files','.cal'),)
        filePath = tk.filedialog.askopenfilename(title= titleStr,
                                                  filetypes=fileTypeTuple)
        
        #open calibration file, extract data, set instance variables
        with open(filePath) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            self.DIMf = tuple([int(i) for i in next(csvReader)])
            K1f = [float(i) for i in next(csvReader)]
            K2f = [float(i) for i in next(csvReader)]
            K3f = [float(i) for i in next(csvReader)]
            self.Kf = np.array([K1f, K2f, K3f])
            D1f = [float(i) for i in next(csvReader)]
            D2f = [float(i) for i in next(csvReader)]
            D3f = [float(i) for i in next(csvReader)]
            D4f = [float(i) for i in next(csvReader)]
            self.Df = np.array([D1f, D2f, D3f, D4f])
            
            self.DIMr = tuple([int(i) for i in next(csvReader)])
            K1r = [float(i) for i in next(csvReader)]
            K2r = [float(i) for i in next(csvReader)]
            K3r = [float(i) for i in next(csvReader)]
            self.Kr = np.array([K1r, K2r, K3r])
            D1r = [float(i) for i in next(csvReader)]
            D2r = [float(i) for i in next(csvReader)]
            D3r = [float(i) for i in next(csvReader)]
            D4r = [float(i) for i in next(csvReader)]
            self.Dr = np.array([D1r, D2r, D3r, D4r])
        
        #update GUI label
        self.msgFrame2["text"] = 'Calibration file successfully loaded' 
        self.msgFrame2.update_idletasks()
        self.camCalLoaded = True
    
    #sets image threshold parameters to default
    def setDefaultParams(self):
        #delete current contents of entry boxes
        self.minHueEnt.delete(0, tk.END)
        self.maxHueEnt.delete(0, tk.END)
        self.minSatEnt.delete(0, tk.END)
        self.maxSatEnt.delete(0, tk.END)
        
        #insert default contents of entry boxes
        self.minHueEnt.insert(0, str(self.minHueDef))
        self.maxHueEnt.insert(0, str(self.maxHueDef))
        self.minSatEnt.insert(0, str(self.minSatDef))
        self.maxSatEnt.insert(0, str(self.maxSatDef))
        
        #disable entry boxes
        self.minHueEnt.config(state=tk.DISABLED)
        self.maxHueEnt.config(state=tk.DISABLED)
        self.minSatEnt.config(state=tk.DISABLED)
        self.maxSatEnt.config(state=tk.DISABLED)
    
    #allows image threshold parameters to be modified
    def setCustParams(self):
        self.minHueEnt.config(state=tk.NORMAL)
        self.maxHueEnt.config(state=tk.NORMAL)
        self.minSatEnt.config(state=tk.NORMAL)
        self.maxSatEnt.config(state=tk.NORMAL)
    
    #launches interactive window to play around with image threshold parameters
    def interactiveThreshold(self):
        #dummy function used in cv2.createTrackbar
        def nothing(var1=None):
            pass
        
        #switch radiobutton from custom to default
        self.custRdBtn.select()
        self.setCustParams()
        
        #ask user for path of image file
        titleStr = 'Select Image To Threshold'
        fileTypeTuple = (('Supported Image Files', '.jpg .png'),)
        filePath = tk.filedialog.askopenfilename(title= titleStr,
                                                  filetypes=fileTypeTuple)
        
        
        # Load in image
        image = cv2.imread(filePath, -1)
        midCol = image.shape[1]//2
        
        for row in range(image.shape[0]):
            if ((row//50) % 2) == 0:
                image[row, midCol-3:midCol+3, :] = np.zeros([6,3])
        image = cv2.resize(image, (700, 400))# Resize image
        
        
        windowName = 'Adjust trackbars to desired threshold, then click exit'
        # Create a window
        cv2.namedWindow(windowName)
        
        # create trackbars for color change
        cv2.createTrackbar('HMin',windowName, 0, 179, nothing) 
        cv2.createTrackbar('SMin',windowName, 0, 255, nothing)
        # cv2.createTrackbar('VMin','image',0,255,nothing)
        cv2.createTrackbar('HMax',windowName, 0, 179, nothing)
        cv2.createTrackbar('SMax',windowName, 0, 255, nothing)
        # cv2.createTrackbar('VMax','image',0,255,nothing)
        
        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', windowName, 179)
        cv2.setTrackbarPos('SMax', windowName, 255)
        # cv2.setTrackbarPos('VMax', 'image', 255)
        
        # Initialize to check if HSV min/max value changes
        #hMin = sMin = vMin = hMax = sMax = vMax = 0
        #phMin = psMin = pvMin = phMax = psMax = pvMax = 0
        
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0
        
        
        output = image
        wait_time = 3
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        while cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) >= 1:
        
            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin',windowName)
            sMin = cv2.getTrackbarPos('SMin',windowName)
            # vMin = cv2.getTrackbarPos('VMin','image')
            vMin = 0
        
            hMax = cv2.getTrackbarPos('HMax',windowName)
            sMax = cv2.getTrackbarPos('SMax',windowName)
            # vMax = cv2.getTrackbarPos('VMax','image')
            vMax = 255
            
            #special Conditioning if hue bridges across 180 barrier
            if hMin >= hMax:
                lower1 = np.array([hMin, sMin, vMin])
                upper1 = np.array([179, sMax, vMax])
                # Create HSV Image and threshold into a range.
                mask1 = cv2.inRange(hsv, lower1, upper1)
                
                lower2 = np.array([0, sMin, vMin])
                upper2 = np.array([hMax, sMax, vMax])
                # Create HSV Image and threshold into a range.
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = (mask1 + mask2)//2
            else:
                # Set minimum and max HSV values to display
                lower = np.array([hMin, sMin, vMin])
                upper = np.array([hMax, sMax, vMax])
            
                # Create HSV Image and threshold into a range.
                mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(image,image, mask=mask)
            
            # Print if there is a change in HSV value
            if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                # print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax
            
            # Display output image
            cv2.namedWindow(windowName)
            cv2.resizeWindow(windowName, int(700), int(650))
            cv2.imshow(windowName,output)
        
        
            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        #write new values to threshold parameter entry boxes
        #delete current contents of entry boxes
        self.minHueEnt.delete(0, tk.END)
        self.maxHueEnt.delete(0, tk.END)
        self.minSatEnt.delete(0, tk.END)
        self.maxSatEnt.delete(0, tk.END)
        
        #insert new values in entry boxes
        self.minHueEnt.insert(0, str(hMin))
        self.maxHueEnt.insert(0, str(hMax))
        self.minSatEnt.insert(0, str(sMin))
        self.maxSatEnt.insert(0, str(sMax))
    
    # generate a list of images that will be used for draft stripe processing
    def loadImages(self):
        #first, ask for directory containing sail images to process
        dialogTitle = 'Select a file folder containing draft stripe images'
        self.imgDirectory = filedialog.askdirectory(title=dialogTitle)
        
        # find all the image files in given directory
        os.chdir(self.imgDirectory)
        images = glob.glob("*.jpg") + glob.glob("*.png")
        
        #return if no images are found in the directory 
        if images == []:
            tk.messagebox.showerror('No Image Error', 'No image files were '
                                    'found. Please check the selected folder'
                                    ' again.')
            return
        
        images.sort(key=os.path.getmtime)
        self.sailImages = images
        self.numImages = len(images)
        self.msgFrame4["text"] = (str(self.numImages) + ' images were loaded')
        self.msgFrame4.update_idletasks()
        self.imagesAreLoaded = True
        
    #run the sail shape program!
    def run(self):
        #check to make sure calibration file loaded
        if not self.camCalLoaded:
            tk.messagebox.showerror('No Calibration File', 'No camera '
                                    'calibration file was loaded. Please load ' 
                                    'one before continuing.')
            return
        
        #check to make sure picture is loaded
        if not self.imagesAreLoaded:
            tk.messagebox.showerror('No Images To Process', 'No sail shape '
                                    'images have been loaded into the program.' 
                                    ' Please choose a directory with your sail'
                                    ' images before continuing.')
            return
        
        #check to make sure valid inputs were given for the threshold params
        try: #first make sure they are integers
            self.minHue = int(self.minHueEnt.get())
            self.maxHue = int(self.maxHueEnt.get())
            self.minSat = int(self.minSatEnt.get())
            self.maxSat = int(self.maxSatEnt.get())
        except:
            tk.messagebox.showerror('Bad Threshold Parameters', 'Image '
                                    'threshold parameters must be integers')
            return
        #check hue and saturation values fall within the correct range
        if not (0 <= self.minHue <= 179 and 0 <= self.maxHue <= 179):
            tk.messagebox.showerror('Bad Hue Range', 'Please check Hue values.'
                                    ' Min Hue must be less than Max Hue and '
                                    'both values should be between 0 and 179')
            return
        if not 0 <= self.minSat < self.maxSat <= 255:
            tk.messagebox.showerror('Bad Saturation Range', 'Please check '
                                    'Saturation values. Min Saturation must be'
                                    ' less than Max Saturation and both values'
                                    ' should be between 0 and 255')
            return
        #package threshold parameters in tuple to be used as input for 
        #Threshold objects
        threshParams = (self.minHue, self.maxHue, self.minSat, self.maxSat,
                        self.minValue, self.maxValue)
        
        # organize calibration data
        self.calibrationData = {}
        reduceBool = self.reduceImageSize.get()
        if reduceBool:
            self.calibrationData['DIM'] = self.DIMr
            self.calibrationData['K'] = self.Kr
            self.calibrationData['D'] = self.Dr
        else:
            self.calibrationData['DIM'] = self.DIMf
            self.calibrationData['K'] = self.Kf
            self.calibrationData['D'] = self.Df
        
        #get number of draft stripes and check if number is acceptable
        try:
            self.numStripes = int(self.numStripesEnt.get())
        except:
            tk.messagebox.showerror('Bad Number Of Stripes', 'Number of '
                                    'stripes should be between 1 and 5')
            return
        if not 0 < self.numStripes < 6:
            tk.messagebox.showerror('Bad Number Of Stripes', 'Number of '
                                    'stripes should be between 1 and 5')
            return
        
        #create an sail threshold object for every sail image loaded
        self.sailThresholds = []
        count = 0
        for imagePath in self.sailImages:
            try:
                sailThreshold = lib.Threshold(imagePath, 
                                              self.calibrationData,
                                              threshParams, 
                                              self.reduceImageSize.get(), 
                                              self.numStripes)
            except:
                print('an error occured while processing image' + 
                      ' {}'.format(imagePath))
                continue
            # update progress bar as images are processed
            count += 1
            progressValue = int(count / self.numImages * 100)
            self.runProgress['value'] = progressValue
            self.window.update_idletasks()
            
            #create directory for overlaid images and add overlaid images 
            if self.overlayImages.get():
                overlayDirectory = self.imgDirectory + '/Data Overlaid Images'
                if not os.path.isdir(overlayDirectory):
                    os.mkdir(overlayDirectory)
                sailThreshold.overlayStripes()
                sailThreshold.overlayData(overlayDirectory)
            
            self.sailThresholds.append(sailThreshold) #save threhsold object
        
        # write outout data to csv file
        os.chdir(self.imgDirectory)
        header = ['Image Name', 'Time Taken']
        for i in range(self.numStripes):
            header.append('Twist, Stripe {}'.format(i+1))
            header.append('Max Draft, Stripe {}'.format(i+1))
            header.append('Draft Location, Stripe {}'.format(i+1))
        
        with open('PSA Output.csv', mode='w', newline='') as outputFile:
            writer = csv.writer(outputFile, delimiter=',')
            writer.writerow(header)
            for sailThreshold in self.sailThresholds:
                writer.writerow(sailThreshold.outputList)
        

if __name__ == '__main__':
    psaGui = GUI()