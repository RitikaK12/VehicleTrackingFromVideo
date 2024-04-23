# VehicleTrackingFromVideo
This script detects and tracks cars in a video using OpenCV's Haar cascade classifier.

How to Run
1.	Install OpenCV: If you haven't already, install OpenCV using pip:
2.	 Download Required Files:
*	Download the haarcascade_car.xml file from here.(link to pretrained haarcascade)
*	Download a video file containing cars. This video has been used for testing 
https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/dataset
/video1.avi
3.	Update File Paths:
*	haarcascade_car.xml path is aa pretrained harrcascade model.
*	the link for the video used for testing has been given above.
4.	Run the Script:
*	I used jupyter notebook to run the script.
5.	View Output:
*	The script will display the video with detected cars outlined in red rectangles.
*	Press 'q' to exit the video.

Explanation
Line by Line explainantion of the code:
*	import cv2: Imports the OpenCV library for computer vision tasks.
*	import ctypes: Imports the ctypes library, which provides C compatible data types and 
allows calling functions in DLLs or shared libraries.
*	vid = cv2.VideoCapture(): Opens a video file specified by the path and creates a 
VideoCapture object vid to capture frames from the video.
*	car_cascade = cv2.CascadeClassifier(r"C:\Users\hii\Downloads\haarcascade_car.xml"): Loads 
a pre-trained Haar cascade classifier for car detection from the specified XML file.
*	user32 = ctypes.windll.user32: Initializes a reference to the Windows user32 library using 
ctypes, which allows accessing functions like GetSystemMetrics to get screen resolution.
*	screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1): 
Retrieves the screen resolution in pixels using GetSystemMetrics function, with 0 for screen 
width and 1 for screen height.
*	cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL): Creates a resizable window 
named 'Object Detection' for displaying the video frames, using the cv2.WINDOW_NORMAL 
flag to allow resizing.
*	cv2.resizeWindow('Object Detection', screen_width, screen_height): Resizes the window to 
match the screen resolution.
*	while True:: Enters a loop to continuously read frames from the video and perform object 
detection.
*	ret, img = vid.read(): Reads a frame from the video capture vid and stores it in the img 
variable. ret is a boolean indicating whether a frame was successfully read.
*	if (type(img) == type(None)):: Checks if the img is None, which can happen if the video 
capture is unsuccessful or if there are no more frames to read. If true, breaks out of the 
loop.
*	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): Converts the BGR color image img to 
grayscale for easier processing.
*	cars = car_cascade.detectMultiScale(gray, 1.1, 1): Detects cars in the grayscale image using 
the Haar cascade classifier car_cascade, with a scale factor of 1.1 and minimum neighbours 
of 1.
*	for (x,y,w,h) in cars:: Iterates over the detected cars, where (x,y) is the top-left corner of the 
bounding box, and (w,h) are the width and height of the bounding box.
*	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2): Draws a red rectangle around each detected 
car on the original color image img.
*	cv2.imshow('Object Detection', img): Displays the image img with the detected cars in a 
window titled 'Object Detection'.
*	if cv2.waitKey(1) & 0xFF == ord('q'):: Waits for a key press for 1 millisecond. If the key 
pressed is 'q' (ASCII value 113), breaks out of the loop.
*	vid.release(): Releases the video capture vid, freeing up the resources associated with it.
*	cv2.destroyAllWindows(): Closes all OpenCV windows.

