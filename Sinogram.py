#Załadowanie niezbędnych bibiotek
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import os

######################################
#         Sekcja konfiguracji        #
######################################

#Nazwa pliku z przekrojem do wczytania
sFileName = "Dots2.png" 
#Nazwa pliku do zapisania sinogramu
sOutFileName = "Sinogram.png"
#Liczba projektcji
AngleSteps = 360
#Debugowanie
bDebug = False
invertedImage = False

Img = cv2.imread(sFileName, cv2.IMREAD_GRAYSCALE)
if invertedImage:
    Img = 255 - Img
(ImgHeight, ImgWidth) = Img.shape

SinogramImg = np.empty((AngleSteps,ImgWidth))

for AngleId in range(0,AngleSteps):
  print("{}/{}".format(AngleId,AngleSteps))
  RotatedImage = ndimage.rotate(Img, 360.0*AngleId/AngleSteps, reshape=False)
  SumInCols = np.sum(RotatedImage,axis=0)
  SinogramImg[AngleId,:] = SumInCols
  if AngleId in range(0,20) and bDebug:
    fig = plt.figure(figsize=(3, 5))
    ax1, ax2 = fig.subplots(2, 1)
    ax1.set_title("Skanowany obiekt")
    if invertedImage:
        RotatedImage = 255 - RotatedImage
    ax1.imshow(RotatedImage,cmap='gray')
    ax2.set_title("Wykonany skan - współczynniki pochłanania promieniowania rentgenowskiego")
    ax2.plot(SumInCols)
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                wspace=0.35)
    plt.show()

n = sum(sum(SinogramImg))
#fig = plt.figure(figsize=(10, 3))
if invertedImage:
    SinogramImg = 255 - SinogramImg

plt.imshow(SinogramImg/n,cmap='gray')
plt.show()
SinogramImg = 255.0*SinogramImg/np.amax(SinogramImg)

cv2.imwrite(sOutFileName,SinogramImg)
