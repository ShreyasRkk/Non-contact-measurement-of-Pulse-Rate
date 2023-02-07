import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
line1=[]
# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


#Webcam Parameters


webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
realWidth = frame.shape[1]
realHeight = frame.shape[0]
# realWidth = 320
# realHeight = 240
# webcam.set(3, realWidth);
# webcam.set(4, realHeight);
#ret, frame = webcam.read()
# frame = cv2.resize(frame, (realWidth, realHeight))
# faces = face_cascade.detectMultiScale(frame)
(x,y,w,h) = (0,0,0,0)
while frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :].size == 0:
    ret, frame = webcam.read()
    # frame = cv2.resize(frame, (realWidth, realHeight))
    faces = face_cascade.detectMultiScale(frame)
    if len(faces) == 0:
        continue
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
# print('Out of first loop..')
# print(faces)
(x,y,w,h) = faces[0]
roi_frame = frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :]

videoWidth = int(roi_frame.shape[1])
videoHeight = int(roi_frame.shape[0])
videoChannels = 3
videoFrameRate = 15


# Output Videos
# if len(sys.argv) != 2:
#     originalVideoFilename = "original.mov"
#     originalVideoWriter = cv2.VideoWriter()
#     originalVideoWriter.open(originalVideoFilename, cv2.cv.CV_FOURCC('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)
#
# outputVideoFilename = "output.mov"
# outputVideoWriter = cv2.VideoWriter()
# outputVideoWriter.open(outputVideoFilename, cv2.cv.CV_FOURCC('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

outputVideoWriter = cv2.VideoWriter('output_harr.avi',cv2.VideoWriter_fourcc('M','J','P','G'), videoFrameRate, (realWidth,realHeight))

# Color Magnification Parameters
levels = 3
alpha = 30
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (255,0,0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
# print("firstFrame shape - " + str(firstFrame.shape) )
firstGauss = buildGauss(firstFrame, levels+1)[levels]
# print("firstGauss shape - " + str(firstGauss.shape) )
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
#print('vidgaushp', np.shape(videoGauss))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

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
    #print('faces_shape', np.shape(faces)) #(1,4) cause [x y w h]
    # print(faces)
    if len(faces) == 0:
        continue

    (x,y,w,h) = faces[0]
    if frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :].size == 0:
        continue
    detectionFrame = cv2.resize(frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :], (videoWidth, videoHeight))
    # print(detectionFrame.shape)
        # detectionFrame2 = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :]
        # print(detectionFrame2.shape)

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels] #screw with levels and see
    #cv2.waitKey(10)
    #print('vidgaushpinlup', np.shape(videoGauss))
    fourierTransform = np.fft.fft(videoGauss, axis=0)
    # print('ft',fourierTransform)
    # print('ft_shape',np.shape(fourierTransform))
    # print("ok", videoGauss[bufferIndex])
    # print('video_gauss_shape', np.shape(videoGauss[bufferIndex]))
    # print('frame_shape', np.shape(detectionFrame))
    # if bufferIndex==2:
    #     break
    # Bandpass Filter
    freq=[]
    fourierTransform[mask == False] = 0
    for ij in frequencies:
        if ij>=minFrequency and ij<=maxFrequency:
            freq.append(ij)

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0: #takes 15th frame and if fps=15, 1 frame/sec= 1Hz(bpmCalculationFrequency)
        i = i + 1
        print(i)
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).max() #why ;-;?
        # print("fft_avg_shape", np.shape(fourierTransformAvg))
        # print("fft_avg", fourierTransformAvg)
        hz = frequencies[np.argmax(fourierTransformAvg)] #max of 150 frames-> corresponding frequency
        bpm = 60.0 * hz


        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize
    # print('buffer index', bufferIndex)

    # detectionFrame = cv2.resize(frame[y:int(y+h*0.25), x+50:int(x+w*0.75), :], (videoWidth, videoHeight)) = outputFrame
    cv2.rectangle(frame, (x+50,y),(int(x+w*0.75),int(y+h*0.25)) , boxColor, boxWeight)
    if i > bpmBufferSize:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

    outputVideoWriter.write(frame)

    if len(sys.argv) != 2:
        outputVideoWriter.write(frame)
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
