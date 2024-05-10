# Vehicle-speed-detection-using-Python-and-OpenCV
The project employs Python and OpenCV to detect vehicle speeds accurately through a systematic methodology. I start by acquiring video footage, defining regions of interest, and applying masking techniques for clarity. Using contour detection, I identify vehicles, enabling precise speed estimation. Upon detection, I capture images and record vehicle data for analysis and monitoring purposes, enhancing road safety and traffic management efficiency.

# Objective
The objective of this projects is:

Vehicle Detection

Speed estimation

Capturing vehicle image

# Vehicle Detection
The vehicle detection code file is instrumental in identifying vehicles within video footage. Through Python and OpenCV, I acquire video inputs and define regions of interest to focus on traffic-relevant areas. Employing masking techniques, I filter out noise and isolate these regions. Subsequently, contour detection algorithms detect vehicle candidates within the masked areas. Visual bounding boxes are then drawn around detected vehicles to validate accuracy.

# Speed Detection
the speed detection code file estimates vehicle speeds. Leveraging techniques like optical flow or frame differencing, I calculate vehicle movement between frames. Data logging functionalities capture essential information such as vehicle IDs and estimated speeds for analysis. Additionally, we capture images of detected vehicles to visually verify speed estimations. 


# PyCharm

Using PyCharm, we can easily run our code to track car speeds. When I run the code, it calculates the speed of cars in the video. The data from this calculation gets saved in a folder named "traffic record." PyCharm makes it simple to run our code and see the results, helping us monitor traffic speed effectively.

