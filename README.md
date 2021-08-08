# Description
Performance Sail Analysis is a python tool created by Richard Didham to
automatically calculate sail shape parameters using color based image 
thresholding.

## Notes Before Using
The program's GUI is fairly self explainatory however a couple things may not
be obvious:
1) You only need to run the calibration routine once. After it finishes,you 
	can load in the calibration file it generates and skip that step assuming
	you are still using the same camera and resolution settings.

2) The status bar at the bottom of the GUI has a tendancy to freeze when 
	the program is running. Keep this in mind if you ever think the program has
	frozen while processing your images.
	
3) When processing red or orange draft stripes, the Hue values bridge from the 
	upper limit (179) back to 0. To account for this, you can set your min hue
	to be a greater value than your max hue and a threshold will be created
	with hue values ranging from minHue to 179 and 0 to maxHue.

4) Feel free to reach out if you have further questions. Especially 3) may be
	difficult to understand without a background in HSV thresholding.