# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:00:52 2021

Version 1.0
@author: Richard Didham. Library of functions used in PSA 

Versions Notes:
    v1.0: initial
"""
import cv2
import csv
import numpy as np
import math
import datetime
import os


class Threshold:
    def __init__(self, imagePath, calibrationData, threshParams, reduceBool,
                 numStripes):
        self.imagePath = imagePath
        self.calibrationData = calibrationData
        self.minHue = threshParams[0]
        self.maxHue = threshParams[1]
        self.minSat = threshParams[2]
        self.maxSat = threshParams[3]
        self.minValue = threshParams[4]
        self.maxValue = threshParams[5]
        self.numStripes = numStripes
        
        rawTimeModified = os.path.getmtime(imagePath)
        dateTimeObjectMod = datetime.datetime.fromtimestamp(rawTimeModified)
        self.timeModified = str(dateTimeObjectMod.time())
        
        #load Image, resize it if user requested, then undistort it
        self.img = cv2.imread(self.imagePath,-1)
        if reduceBool:
            self.img = cv2.resize(self.img, (int(self.img.shape[1]/4),
                                             int(self.img.shape[0]/4)))
        self.undistort() 
        
        # Create and threshold HSV
        self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # lowerBound = np.array([self.minHue, self.minSat, self.minValue])
        # upperBound = np.array([self.maxHue, self.maxSat, self.maxValue])
        # self.mask = cv2.inRange(self.hsvImg, lowerBound, upperBound)
        # self.searchCol = self.mask.shape[1]//2 #middle Column
        
        #special conditioning if hue bridges across 180 border
        if self.minHue >= self.maxHue: 
            lowerBound1 = np.array([self.minHue, self.minSat, self.minValue])
            upperBound1 = np.array([179, self.maxSat, self.maxValue])
            # Create HSV Image and threshold into a range.
            mask1 = cv2.inRange(self.hsvImg, lowerBound1, upperBound1)
            
            lowerBound2 = np.array([0, self.minSat, self.minValue])
            upperBound2 = np.array([self.maxHue, self.maxSat, self.maxValue])
            # Create HSV Image and threshold into a range.
            mask2 = cv2.inRange(self.hsvImg, lowerBound2, upperBound2)
            self.mask = (mask1 + mask2)//2
        else:
            lowerBound = np.array([self.minHue, self.minSat, self.minValue])
            upperBound = np.array([self.maxHue, self.maxSat, self.maxValue])
            self.mask = cv2.inRange(self.hsvImg, lowerBound, upperBound)
        self.searchCol = self.mask.shape[1]//2 #middle Column
        
        self.findStripes()
        self.calcData()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Undistorts image using calibration data from the calibrate() method
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def undistort(self):
        DIM = self.calibrationData['DIM']
        K = self.calibrationData['K']
        D = self.calibrationData['D']
        balance = 0
        dim1 = self.img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        dim2 = dim1
        dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistortedImg = cv2.remap(self.img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        self.img = undistortedImg
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Looks through thresholded image and tries to find the draft stripes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def findStripes(self):
        [numRows, numCols] = self.mask.shape #find the dimentions of the image
        
        #search for tapelines by starting in middle colunm and walking down
        self.midStripeRows = []
        self.thicknesses = []
        count = 0
        possibleStart = 0
        self.centroidsDict = {}
        self.glareFlags = {}
        lastMidStripeRow = 0
    
          
        countingRows = False      
        for i in range(numRows-10): #check the value of the middle index of every row
            if all([self.mask[i, self.searchCol] != 0, not countingRows,
                    i > (lastMidStripeRow+25)]): #this indicates the top row of a flag
                possibleStart = i
                countingRows = True
            
            elif self.mask[i, self.searchCol] == 0 and countingRows: # atleast 1 row of 255s has been detected
                countingRows = False #this means we are no longer counting 255s
                possibleFinish = i
                thickness = possibleFinish - possibleStart #this is the thickness of the potential tapleine
                if thickness > 3: #the thicness must be greater than 4 to be considered a potential tapeline
                    midStripeRow = int(possibleFinish)
                    
                    #[X,Y] CONVENTION! right or left centroids starting from middle
                    rightCents = self.search(midStripeRow, self.searchCol, self.mask,
                                        'right', thickness) 
                    leftCents = self.search(midStripeRow, self.searchCol, self.mask,
                                       'left', thickness) 
                    
                    #if the number of rows in the output centroid file is too small, then this row flag is too small to be considered a legit tapeline
                    if leftCents.shape[0] + rightCents.shape[0] > thickness*25: 
                        self.midStripeRows.append(midStripeRow)
                        self.thicknesses.append(thickness)
                        
                        #[X,Y] CONVENTION! #Combined the left and right centroid arrays together and ordered the numbers from left to right
                        combinedCents = np.append(leftCents[::-1,:], 
                                                  rightCents[1:,:], axis=0) 
                        #array used to edit values
                        editedCents = np.full([combinedCents.shape[0], 1],
                                              numRows-1) 
                        #convert centroid cooridnates from picture convention into cartesian coordinates
                        cartesianCents = np.append(combinedCents[:,:1],
                                                   editedCents-combinedCents[:,1:2],
                                                   axis=1)
                        #change to integer for array indexing and store in dictionary
                        self.centroidsDict[count] = cartesianCents.astype(int)              
                        
                        rec = rightCents[-1,:].astype(int) # right end centroid used to check if glare was a problem
                        lec = leftCents[-1,:].astype(int) # left end centroid used to check if glare was a problem
                        
                        if sum(self.hsvImg[rec[1]-5:rec[1]+5,rec[0],1])<10*20:
                            self.glareFlags[count] = True
                        elif sum(self.hsvImg[lec[1]-5:lec[1]+5,lec[0],1])<10*20:
                            self.glareFlags[count] = True
                        else:
                            self.glareFlags[count] = False
                        lastMidStripeRow = midStripeRow
                        count += 1
                if count==self.numStripes: #this is the goal nunber of tapes
                    break
                possibleStart = 0
        
        self.numStripesFound = count
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Given the coodinates of the draft stripes, calcData() calculates the
    # draft location, depth, and twist of each stripe
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def calcData(self):
        self.outputDict = {}
        self.outputList = [self.imagePath, self.timeModified]
        ##############  calculate twist, depth and chamber for 3 tapelines ############
        for i in range(self.numStripesFound):
            centroids = self.centroidsDict[i]
            self.outputDict[i] = []
            
            ### calculate vector between two end points of tapeline
            lEnd = centroids[0,:] #[column,row] left most point #[X,Y] CONVENTION!
            rEnd = centroids[-1,:] #[column,row] right most point #[X,Y] CONVENTION!
            dist = np.zeros([centroids.shape[0]-2])
            xl = lEnd[0] #left end, x coordinate
            yl = lEnd[1] #left end, y coordinate
            xr = rEnd[0] #right end, x coordinate
            yr = rEnd[1] #right end, y coordinate
            chordLen = math.sqrt((xl-xr)**2+(yl-yr)**2) #total chord length for the current tapeline
            m1 = (yl-yr)/(xl-xr) # slope of the chordline
            if m1 == 0: #if the slope of the  chordline is zero, then depth calc is simply difference in y coordinates
                dist = yl - centroids[1:(centroids.shape[0]-1), 1]
            else: #if the slope of the line between the two endpoints is not zero, 
                  #then the depth calc requires a projection via a line perpendicular to m1
                m2 = -1/m1 # slope of line perpendicular to the chordline
                count = 0
                #loop through tapeline coordinates to find the delth corresponding to every pixel of the tapeline
                for j in range(1,centroids.shape[0]-1): 
                    x = (m1*xl-yl - m2*centroids[j,0] + centroids[j,1]) / (m1 - m2)
                    y = m1*(x-xl)+yl
                    dist[count] = math.sqrt((x-centroids[j,0])**2 + (y-centroids[j,1])**2) #distance for chamber measument
                    count += 1
            
            maxDist = np.amax(dist)
            
            twist = -1*math.degrees(math.atan((yr-yl)/(xr-xl)))
            self.outputDict[i].append(round(twist, 2)) #twist
            self.outputList.append(twist)
            
            draftDepth = maxDist / chordLen
            self.outputDict[i].append(round(draftDepth, 3)) #max camber
            self.outputList.append(draftDepth)
            
            draftLocation = (np.argmax(dist)+1) / (centroids.shape[0])
            self.outputDict[i].append(round(draftLocation, 2)) #draft location
            self.outputList.append(draftLocation)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # when a potential draft stripe is found, search() walks along the stripe 
    # in each direction to find the coordinates
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def search(self, row_flag, col_flag, mask,direction, thickness): #TC stands for Tape coordinate. this is a nx2 array holding coordinates ([X,Y] convention) of the tapeline
        
        initial_size=int(mask.shape[1]*3/5)
        TC=np.zeros([initial_size,2]) #empty array to store tapeline coordinates. [X,Y] CONVENTION!
        TC[0,0]=col_flag #initialize starting column
        TC[0,1]=row_flag #initialize starting row
        
        
        if direction=='left' or direction=='Left':
            sign=-1  #things will be decreasing
            while_check=0
            col_guess=mask.shape[1]
        elif direction=='right' or direction=='Right':
            sign=1
            while_check=mask.shape[1]
            col_guess=0
        else:
            assert False, 'direction for line_search must be specified as strings: "left" or "right" '
        
        count=1
        #adj_row_rng=int(round(13*mask.shape[0]/612)) #range of rows to check
        #adj_row_rng=int(row_thickness)
        adj_row_rng=3
        search_lim=int(round(3*mask.shape[1]/816)) #limit on the number of empty columns to search after the tapeline has appeared to end
        while sign*(col_guess+sign*search_lim)<sign*while_check: #keep moving in direction until the tapleine ends or you reach the edge of the picture
            col_guess=int(TC[count-1,0]+sign*1) #move over 1 column in the search direction to check if the tapleline continues
            row_guess=int(TC[count-1,1]-thickness/2) #check the same row as the last confirmed entry to check if the tapleline continues
            for i in range(search_lim):  #for loop is used to check a couple additional columns incase there is a small gap in the tapeline columns. It breaks when it finds the next column containing a 255 within the search criteria.
                if sum(mask[row_guess-adj_row_rng:row_guess+adj_row_rng+1,col_guess+sign*i])>0:
                    col_guess=col_guess+sign*i #shift over i number of columns from the guess that was made before the for loop. most of the time, exit criteria for the for loop will be met when i=0
                    guess_range=mask[row_guess-adj_row_rng:row_guess+adj_row_rng+1,col_guess]
                    nz_indexes=np.nonzero(guess_range)[0]
                    index_of_cent_index=(np.abs(nz_indexes - adj_row_rng)).argmin()
                    cent_index=nz_indexes[index_of_cent_index]
                    top_row=int(cent_index+row_guess-adj_row_rng)
                    bottom_row=top_row
    
                    while mask[top_row,col_guess]!=0: #look up until you find the top of the tapeline (note the 'row' or 'y' axis is flipped upside down for pictures)
                        top_row=top_row-1
                    while mask[bottom_row,col_guess]!=0:#look down until you find the bottom of the tapeline (note the 'row' or 'y' axis is flipped upside down for pictures)
                        bottom_row=bottom_row+1
                    thickness=bottom_row-top_row
                    TC[count,0]=col_guess #record the column for the next entry in the tapeline
                    TC[count,1]=int(bottom_row) #the bottom row coordinate of the tapeline is recorded
                    count=count+1
                    break #this breaks the for loop, in turn the while loop will look for the next entry for SC
                if i==search_lim-1: #This if statememnt checks if the search limit hs been reached. the '-1' is a result of 0 indexing wth the range function.
                    col_guess=while_check  #this forces the while loop to break
        TC=TC[~np.all(TC == 0, axis=1)] #Remove unused array indexes
        return(TC) #should be good to go, just need to confirm that multiple outputs can be returned
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #overlays the calculated sail shape parameters on the sail
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def overlayData(self, overlayDirectory):
    #def overlay(img,output):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 3
        
        cv2.putText(self.img, 'Twist', (250,80), font, fontSize, (0,0,0), 2)
        cv2.putText(self.img, 'Depth', (600,80), font, fontSize, (0,0,0), 2)
        cv2.putText(self.img, 'Dft Loc', (1000,80), font, fontSize, (0,0,0), 2)

        # cv2.putText(self.img, 'Top:', (10,200), font, fontSize, (0,0,0), 2)
        # cv2.putText(self.img, 'Mid:', (10,300), font, fontSize, (0,0,0), 2)
        # cv2.putText(self.img, 'Bot:', (10,400), font, fontSize, (0,0,0), 2)
        
        for i in range(self.numStripesFound):
            cv2.putText(self.img, '#{}:'.format(i + 1), (10, 200+i*100),
                        font, fontSize, (0,0,0), 2)
            twist = self.outputDict[i][0]
            depth = self.outputDict[i][1]
            depthLocation = self.outputDict[i][2]
            cv2.putText(self.img, str(twist), (250, 200+i*100), font, fontSize,
                        (0,0,0), 2)
            cv2.putText(self.img, str(depth), (600, 200+i*100), font, fontSize,
                        (0,0,0), 2)
            cv2.putText(self.img, str(depthLocation), (1000, 200+i*100), font,
                        fontSize, (0,0,0), 2)
        
        writePath = overlayDirectory + '/Data Overlaid ' + self.imagePath
        cv2.imwrite(writePath, self.img)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # modifies the image to show the draft stripes that were found
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def overlayStripes(self):
        for value in self.centroidsDict.values():
            for i in range(value.shape[0]): #[X,Y] CONVENTION for centroid arrays
                col = value[i,0]
                row = -value[i,1]
                self.img[row-3:row+3,col,:] = np.full([6, 3], 255, dtype=int)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # modifies the image to show the image column that was walked through when
    # while searching for draft stripes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def overlaySearchLine(self):
        for row in range(self.img.shape[0]):
            self.img[row, self.searchCol-3:self.searchCol+3,:] = np.full([6,3],
                                                                         255,
                                                                         dtype=int)

"""
Calibrate my camera to remove fish-eye distrotion. This file is the same as the feb20
fisheye clibration file except it calculates camera parameters for the original 
size images and the low res version of the images
"""
def calibrate(images):
    #preliminary variables
    CHECKERBOARD = (6,9) #dimentions of the checkerboard -(1,1) (due to 0 indexing)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in images:
        img = cv2.imread(fname) #open image file
        if _img_shape == None:
            _img_shape = img.shape[:2] #find the shape of the image [rows,columns]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #greyscale the image
        if _img_shape[0]>1500: #this means the picture may be large enough that find chessboard corners may fail
            scale=_img_shape[0]/1500
            img_sm = cv2.resize(img, (int(img.shape[1]/scale),int(img.shape[0]/scale))) # Resize image
            gray_sm = cv2.cvtColor(img_sm,cv2.COLOR_BGR2GRAY) #greyscale the image
            # Find the chess board corners
            ret, corners_sm = cv2.findChessboardCorners(gray_sm, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners=corners_sm*scale
        else: #the image is not that large and find chessboard corners should work
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            #imgpoints.append(corners)
            imgpoints.append(corners.reshape(1,-1,2))
    N_OK = len(objpoints)
    D1 = np.zeros((4, 1))
    K1 = np.zeros((3, 3))
    rvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(objpoints,
            imgpoints,
            gray.shape[::-1],
            K1,
            D1,
            rvecs1,
            tvecs1,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM1=_img_shape[::-1]
    
    CHECKERBOARD = (6,9) #dimentions of the checkerboard -(1,1) (due to 0 indexing)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for fname in images:
        img = cv2.imread(fname) #open image file
        img = cv2.resize(img, (int(img.shape[1]/4),int(img.shape[0]/4))) # Resize image
        if _img_shape == None:
            _img_shape = img.shape[:2] #find the shape of the image [rows,columns]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #greyscale the image
        if _img_shape[0]>1500: #this means the picture may be large enough that find chessboard corners may fail
            scale=_img_shape[0]/1500
            img_sm = cv2.resize(img, (int(img.shape[1]/scale),int(img.shape[0]/scale))) # Resize image
            gray_sm = cv2.cvtColor(img_sm,cv2.COLOR_BGR2GRAY) #greyscale the image
            # Find the chess board corners
            ret, corners_sm = cv2.findChessboardCorners(gray_sm, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners=corners_sm*scale
        else: #the image is not that large and find chessboard corners should work
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            #imgpoints.append(corners)
            imgpoints.append(corners.reshape(1,-1,2))
    N_OK = len(objpoints)
    D2 = np.zeros((4, 1))
    K2 = np.zeros((3, 3))
    rvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(objpoints,
            imgpoints,
            gray.shape[::-1],
            K2,
            D2,
            rvecs2,
            tvecs2,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM2=_img_shape[::-1]       
        
        
    # print(str(N_OK) + " valid images were able to be used for calibration." +
    #       " Enter 'y' to save the calibration data or enter 'n' to exit the program")
    # save=input()
    save = 'y'
    if save=="Y" or save=="y":
        with open('PSA_calibration.cal', 'w', newline='') as csvfile:
            K1=K1.tolist()
            D1=D1.tolist()
            K2=K2.tolist()
            D2=D2.tolist()
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(DIM1)
            writer.writerow(K1[0])
            writer.writerow(K1[1])
            writer.writerow(K1[2])
            writer.writerow(D1[0])
            writer.writerow(D1[1])
            writer.writerow(D1[2])
            writer.writerow(D1[3])
            
            writer.writerow(DIM2)
            writer.writerow(K2[0])
            writer.writerow(K2[1])
            writer.writerow(K2[2])
            writer.writerow(D2[0])
            writer.writerow(D2[1])
            writer.writerow(D2[2])
            writer.writerow(D2[3])

    
    return N_OK

if __name__ == '__main__':
    """
    #vrib test
    calibrationData = {'DIM': (4000, 3000)}
    calibrationData['K'] = np.array([[1.76776280e+03, 0.00000000e+00, 2.00316133e+03],
                                     [0.00000000e+00, 1.74602376e+03, 1.48814179e+03],
                                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    calibrationData['D'] = np.array([[-0.03073368],
                                     [ 0.22588899],
                                     [-0.25260214],
                                     [ 0.10536848]])
    """
    """
    #gopro test
    calibrationData = {'DIM': (3264,2448)}
    calibrationData['K'] = np.array([[1383.8949075815146,0.0,1639.934751914607],
                                     [0.0,1374.3830266826335,1241.487784013595],
                                     [0.0,0.0,1.0]])
    calibrationData['D'] = np.array([[0.12266290773671186],
                                     [-0.07653064931065],
                                     [0.06952437073892835],
                                     [-0.029402901937756458]])    
    """
    #helena test
    calibrationData = {'DIM': (2592,1944)}
    calibrationData['K'] = np.array([[1230.1703227376993,0.0,1353.3321042193472],
                                     [0.0,1221.8392433329377,1010.3899894228905],
                                     [0.0,0.0,1.0]])
    calibrationData['D'] = np.array([[-0.04089886763197215],
                                     [0.23524357495046017],
                                     [-0.39179308253568135],
                                     [0.21377627582356581]])
    
    threshParams = [45, 90, 25, 255, 0, 255]
    reduceImageSize = False
    numStripes = 3
    
    #threshold = Threshold('VIRB0210-14.JPG', #virb test
    #threshold = Threshold('G0020335.JPG', #gopro test
    threshold = Threshold('G0022312.JPG', #helena sail
                          calibrationData,
                          threshParams, 
                          reduceImageSize, 
                          numStripes)
    threshold.overlayData(r'C:\Users\rdidh\Documents\Moth_Equipment\PSA\GUI Experiments\Single Image Test')
    
    threshold.overlayStripes()
    threshold.overlaySearchLine()
    cv2.imshow('image', cv2.resize(threshold.img,(int(threshold.img.shape[1]/4),
                                                  int(threshold.img.shape[0]/4))))
    cv2.imshow('mask', cv2.resize(threshold.mask,(int(threshold.mask.shape[1]/4),
                                                 int(threshold.mask.shape[0]/4))))
    
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows() 