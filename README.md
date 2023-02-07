# Non-contact-measurement-of-Pulse-Rate

An implementation of Eulerian video magnification using computer vision. This program uses the method for the application of remotely detecting an individual's heart rate in beats per minute from a still video of his/her face.

Built with OpenCV, NumPy, and SciPy in Python 3

## Program organization:
The evm_face and the ev_forehead files contains the main program that utilizes all of the following modules defined below
to read in the input video, run Eulerian magnification on it, and to display the results. The organisation of the codes are described below:
- Input video preprocessing- To read in video from file and uses Haar cascade face detection to select an ROI on all frames
- pyramids generation - To generate and collapse image/video pyramids (Gaussian/Laplacian)
- Filtering - temporal bandpass filter that uses a Fast-Fourier Transform
- Calculate heart rate from FFT results.

## Inorder to run the code:
 - Type python_face.py to calculate pulse rate using face as the Region Of Interest
 - Type python_forehead.py to calculate pulse rate using forehead as Region of Interest.
  
