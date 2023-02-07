import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
# def reconstructFrame(pyramid, index, levels):
#     filteredFrame = pyramid[index]
#     for level in range(levels):
#         filteredFrame = cv2.pyrUp(filteredFrame)
#     filteredFrame = filteredFrame[:videoHeight, :videoWidth]
#     return filteredFrame

# Webcam Parameters

webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
realWidth = frame.shape[1]
realHeight = frame.shape[0]
# realWidth = 320
# realHeight = 240
# webcam.set(3, realWidth);
# webcam.set(4, realHeight);
ret, frame = webcam.read()
# frame = cv2.resize(frame, (realWidth, realHeight))
# faces = face_cascade.detectMultiScale(frame)
(x,y,w,h) = (0,0,0,0)
while frame[y:y+h, x:x+w, :].size == 0:
    ret, frame = webcam.read()
    # frame = cv2.resize(frame, (realWidth, realHeight))
    faces = face_cascade.detectMultiScale(frame)
    if len(faces) == 0:
        continue
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
#print('Out of first loop..')
#print(faces)
(x,y,w,h) = faces[0]
print('faces[0]', faces[0])
roi_frame = frame[y:int(y+h), x+50:int(x+w), :]
cv2.imshow('roi_frame', roi_frame)
#print("roi_frame", roi_frame)
videoWidth = int(frame.shape[1])
videoHeight = int(frame.shape[0])
videoChannels = 3
videoFrameRate = 15

# Output Videos
# if len(sys.argv) != 2:
#     originalVideoFilename = "original.mov"
#     originalVideoWriter = cv2.VideoWriter()
#     originalVideoWriter.open(originalVideoFilename, cv2.cv.CV_FOURCC('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)
# outputVideoFilename = "output.mov"
# outputVideoWriter = cv2.VideoWriter()
# outputVideoWriter.open(outputVideoFilename, cv2.cv.CV_FOURCC('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

outputVideoWriter = cv2.VideoWriter('output_harr.avi',cv2.VideoWriter_fourcc('M','J','P','G'), videoFrameRate, (realWidth,realHeight))

# Color Magnification Parameters
levels = 3
#alpha = 32
minFrequency = 1.0
min_new = 0.5
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
bpmTextLocation1 = (videoWidth//2 + 5, 60)
fontScale = 1
fontColor = (255,0,0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

##Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
#print("firstFrame shape - " + str(firstFrame.shape) )
firstGauss = buildGauss(firstFrame, levels+1)[levels]

#print("firstGauss shape - " + str(firstGauss.shape) )
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
freq = (1.0*videoFrameRate) * np.arange(bufferSize) / (10.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 20
bpmBuffer = np.zeros((bpmBufferSize))

plt.ion() # Stop matplotlib windows from blocking

# Setup figure, axis and initiate plot
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro-')

i = 0
while (True):
    ret, frame = webcam.read()
    # frame = cv2.resize(frame, (realWidth, realHeight))
    if ret == False:
        break

    # if len(sys.argv) != 2:
    #     originalFrame = frame.copy()
    #     originalVideoWriter.write(originalFrame)

    faces = face_cascade.detectMultiScale(frame)
    #print("faces_shape", np.shape(faces))
    if len(faces) == 0:
        continue

    (x,y,w,h) = faces[0]
    if frame[y:y+h, x:x+w, :].size == 0:
        continue
    detectionFrame = cv2.resize(frame, (videoWidth, videoHeight))
    #print(detectionFrame.shape)
        # detectionFrame2 = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :]
        # print(detectionFrame2.shape)

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)
    #print("vid_gauu", videoGauss[bufferIndex])
    #print("vid_gau_sp", np.shape(videoGauss))

    # Bandpass Filter
    fourierTransform[mask == False] = 0

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).max()
           
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        time.sleep(0.5)
        xdata = frequencies
        ydata = fourierTransformAvg
        ln.set_xdata(xdata)
        ln.set_ydata(ydata)
        # Rescale the axis so that the data can be seen in the plot
        # if you know the bounds of your data you could just set this once
        # so that the axis don't keep changing
        ax.relim()
        ax.autoscale_view()

        # Update the window
        fig.canvas.draw()
        fig.canvas.flush_events()
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
        #print("bpmBufferIndex", bpmBufferIndex)

    # Amplify
    # filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    # filtered = filtered * alpha

    # Reconstruct Resulting Frame
    # filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    # outputFrame = detectionFrame + filteredFrame
    # outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize
    #print("bufferIndex", bufferIndex)
    temp=37.64
    # detectionFrame = cv2.resize(frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :], (videoWidth, videoHeight)) = outputFrame
    cv2.rectangle(frame,(x,y),(int(x+w),int(y+h)), boxColor, boxWeight)
    if i > bpmBufferSize:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
        #cv2.putText(frame, "Temp: %.2f" % temp, bpmTextLocation1, font, fontScale, boxColor, lineType)
    else:
        cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)


    outputVideoWriter.write(frame)


    cv2.imshow("Webcam Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()