from handTracking import *
import cv2
# import mediapipe as mp
import numpy as np
import random
import time
import os
import Point
import Function
from model import SimpleNet
import torch
import pandas as pd
import csv
from PIL import ImageFont, ImageDraw, Image



class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha = 0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text=text
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        #draw the box
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
        # Putting the image back to its position
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        #put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)

    def isOver(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False



#initilize the habe detector
detector = HandTracker(detectionCon=0.8)



# creating canvas to draw on it
canvas = np.zeros((720,1280,3), np.uint8)

# define a previous point to be used with drawing a line
px,py = 0,0
#initial brush color
color = (255,255,255)
#####
brushSize = 10
eraserSize = 20
iter = 1
#Define Coordinate and save file
folder = "CoorData"
coordinates = []
khmer_char_predicted = []


dataPath = "CoorData"
filename = "1.txt"
ScaleDataPath = "./PredictData/scaledata3.csv"
Path = f"{dataPath}/{filename}"
uniPoint = []
numberOfColumnOfEachChar = []

if not os.path.exists(folder):
    os.makedirs(folder)
coor_files = len([f for f in os.listdir(folder) if f.startswith('coordinates')])
filename = f"{folder}/{coor_files+1}.txt"


colors = []


########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(800+i*100,0,100,100, (0,0,0), str(penSize)))



#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1060, 580, (255,255,255),alpha = 0.6)

coolingCounter = 20
hideBoard = False
hideColors = False
hidePenSizes = True

#initilize the camera 
cam_index = 0
cap = cv2.VideoCapture(cam_index)
cap.set(3, 1280)
cap.set(4, 720)

while True:

    if coolingCounter:
        coolingCounter -=1
        #print(coolingCounter)

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    detector.findHands(frame)
    positions = detector.getPostion(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers:
        
        x, y = positions[8][0], positions[8][1]
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0


            ##### pen sizes ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            # ###### chose a color for drawing #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

    
        elif upFingers[1] and not upFingers[2] and not upFingers[3] and  not upFingers[4]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                # print('index finger is up')
                cv2.circle(frame, positions[8], brushSize, color,-1)
                #drawing on the canvas
                if px == 0 and py == 0:
                    # print("Hello")
                    px, py = positions[8]
                    
                if color == (0,0,0):
                    cv2.line(canvas, (px,py), positions[8], color, eraserSize)
                else:
                    cv2.line(canvas, (px,py), positions[8], color,brushSize)
                px, py = positions[8]

                # print((px, py))
                coordinate = (px, py)
                coordinates.append(coordinate)

            # print(coordinates)

        
        elif upFingers[0] and not  upFingers[1]:
            coordinates = [] 
            # clear.alpha = 0
            # px, py = 0, 0
            canvas = np.zeros((720,1280,3), np.uint8)
            clr_file = open(filename, "r+")
            clr_file.truncate(0)
            clr_scaledfile = open(ScaleDataPath, "r+")
            clr_scaledfile.truncate(0)

        elif not upFingers[1] and not upFingers[2] and  not upFingers[3] and  not upFingers[4] and not  upFingers[0]:
            
            print("Thank you..............................")
            break
    
        # elif upFingers[1] and upFingers[2] and not upFingers[3]:
        elif upFingers[1] and upFingers[2] and  upFingers[3] and  upFingers[4] and not  upFingers[0]:
            with open(filename, "w") as file:
            # Loop through the coordinates and write each one to the file
                for x, y in coordinates:
                    file.write(f"{x} {y} ")

            if filename[0] != ".":          
                f = open(Path,  mode="r", encoding='windows-1252')
                data = f.read()

                if data != "":
                    strokes = data.split(("\n"))

                    allStroke = []
                    allxPoint = []
                    allyPoint = []


                    csvRow = []

                    allStroke = Function.cleanEmptyStroke(allStroke)

                    for stroke in strokes:
                        rawPointsInStroke = stroke.split((" "))
                        pointInStoke = []
                        x = 0
                        y = 0
                        for i in range(len(rawPointsInStroke)-1):
                            if i%2 == 0:
                                x = int(rawPointsInStroke[i])
                                allxPoint.append(x)

                            if i%2 != 0:
                                y = int(rawPointsInStroke[i])
                                allyPoint.append(y)

                                point = Point.Point(x, y)
                                pointInStoke.append(point)

                        allStroke.append(pointInStoke)


                    xMin = min(allxPoint)
                    yMin = min(allyPoint)

                    xMax = max(allxPoint)
                    yMax = max(allyPoint)

                    width = xMax - xMin
                    height = yMax - yMin

                    allNewData = []
                    for stroke in allStroke:
                        newStroke = []
                        for beforePoint in stroke:
                            x = beforePoint.x-xMin
                            y = beforePoint.y-yMin

                            # print(x)

                            if width or width > 0:
                                x /= width
                                y /= height


                            elif width or width == 0:                   
                                print("Division by zero is not allowed. Please Try Again")

                            # print ("X: ",x,"\nY:", y)
                            point = Point.Point(x, y)
                            newStroke.append(point)

                            csvRow.append(x)
                            csvRow.append(y)

                            allNewData.append(newStroke)
                        uniPoint.append(csvRow)
                        numberOfColumnOfEachChar.append(len(csvRow))

        # maxColumn = max(numberOfColumnOfEachChar)
            maxColumn = 2362
            numberOfRow = 1

            with open(file=ScaleDataPath, mode="w") as data:
                for row in uniPoint:
                    numCol = len(row)
                    numberNeedToFill = 1
                    numberOfValue = 1
                    if numCol < maxColumn:

                        for i in range(numCol-1, maxColumn-1):
                            row.append(0)
                            numberNeedToFill = 1

                        for value in row:
                            data.write(f"{value},")
                
                        data.write("\n")
                 

            with open('PredictData/scaledata3.csv', mode="r") as data:
                
                reader = csv.reader(data)
                try:
                    rows = list(reader) 
                except StopIteration:
                    rows = []  


            data = pd.read_csv('datasets/train_data.csv', header=None).to_numpy()
            x = data[:, 1:]
            in_sz = x.shape[1]
            layers_sz = [1024, 512]
            out_sz = 33

            PATH = "save/savedModel"
            net = SimpleNet(layers_sz, in_sz, out_sz)

            loadModel = net.load_state_dict(torch.load(PATH))

            if len(rows) > 0:
                data_test = pd.read_csv('PredictData/scaledata3.csv', header=None).to_numpy()
                x_test = torch.tensor(data_test[0][:-1], dtype=torch.float32)
                # print(x_test)
                net.eval()
                z_test = net(x_test)
                # net.eval()
                # z_test = net(x_test)

                # print(z_test)

                # output = torch.argmax(z_test)
                # print(output)

                # print(z_test)
                output_index = torch.argmax(z_test).item()
                khmer_char = {0: 'ក', 1: 'ខ', 2: 'គ', 3: 'ឃ', 4: 'ង', 5: 'ច', 6: 'ឆ', 7: 'ជ', 8: 'ឈ', 9: 'ញ', 10: 'ដ', 11: 'ឋ', 12: 'ឌ', 13: 'ឍ', 14: 'ណ', 15: 'ត', 16: 'ថ', 17: 'ទ', 18: 'ធ', 19: 'ន', 20: 'ប', 21: 'ផ', 22: 'ព', 23: 'ភ', 24: 'ម', 25: 'យ', 26: 'រ', 27: 'ល', 28: 'វ', 29: 'ស', 30: 'ហ', 31: 'ឡ', 32: 'អ'}
                khmer_char_predicted = khmer_char.get(output_index)
                
                # output = torch.argmax(z_test)
                # print(output_index) 
                
                print(khmer_char_predicted)

            if len(rows) == 0:
                print("No Data to Predict, Draw Again!!")

        else: 
            px, py = 0, 0
              
        
    
    #put the white board on the frame
    if not hideBoard:       
        whiteBoard.drawRect(frame)
        ########### moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)


    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x +pen.w, pen.y+pen.h), (255,255,255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
  
    
    fontpath_limon = "/Users/yornvanda/Documents/Research_I5/Research_Project/Airwriting/fonts/limonf3.TTF" 
    font_kh_limon = ImageFont.truetype(fontpath_limon, 200)
    
    text = "GkSr"
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    kh_text = draw.text( (150, -75),text , font = font_kh_limon, fill = color)
    frame = np.array(img_pil)

    # cv2.putText(frame,  kh_text , (350, -50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1, cv2.LINE_AA)



    fontpath = "/Users/yornvanda/Documents/Research_I5/Research_Project/Airwriting/fonts/KhmerOS.ttf" 
    font_kh = ImageFont.truetype(fontpath, 128)

    # b,g,r,a = 0,255,0,0
    khmer_char_str = ''.join(khmer_char_predicted)
    # khmer_char_str = "ក"
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    kh_text = draw.text( (550, -75), khmer_char_str , font = font_kh, stroke_width=2, fill = color)
    frame = np.array(img_pil)
    # cv2.putText(frame,  kh_text, (350, 0), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1, cv2.LINE_AA)
    # cv2.putText(frame, khmer_char_str, (550, 50), font , 2, (255,255,255), 4)
    

    Start = cv2.imread('Finger_Image/Start.png')
    Draw = cv2.imread('Finger_Image/Draw.jpeg')
    Clear = cv2.imread('Finger_Image/Clear.jpg')
    Save = cv2.imread('Finger_Image/Save.jpeg')
    Close = cv2.imread('Finger_Image/Close.jpg')

    # Resize the image to fit the frame
    resized_image_start = cv2.resize(Start, (120, 120))
    resized_image_draw = cv2.resize(Draw, (120, 120))
    resized_image_clear = cv2.resize(Clear, (120, 120))
    resized_image_save = cv2.resize(Save, (120, 120))
    resized_image_close = cv2.resize(Close, (120, 120))

    
    # fontScale
    fontScale = 1
    

    # Line thickness of 2 px
    thickness = 2

    # Color 
    color =  (255, 0, 0)
    
    # Put text in Images
    resized_image_start = cv2.putText(resized_image_start, 'Start', (25,70), font, fontScale, color, thickness, cv2.LINE_AA)
    resized_image_draw = cv2.putText(resized_image_draw, 'Draw', (25,70), font, fontScale, color, thickness, cv2.LINE_AA)
    resized_image_clear = cv2.putText(resized_image_clear, 'Clear', (25,70), font, fontScale, color, thickness, cv2.LINE_AA)
    resized_image_save = cv2.putText(resized_image_save, 'Save', (25,70), font, fontScale, color, thickness, cv2.LINE_AA)
    resized_image_close = cv2.putText(resized_image_close, 'Close', (25,70), font, fontScale, color, thickness, cv2.LINE_AA)

    # Display the image on the frame
    frame[30:150, 1130:1250] = resized_image_start 
    frame[160:280, 1130:1250] = resized_image_draw
    frame[290:410, 1130:1250] = resized_image_clear 
    frame[420:540, 1130:1250] = resized_image_save
    frame[550:670, 1130:1250] = resized_image_close
    

    cv2.imshow('video', frame)
    # cv2.imshow('canvas', canvas)
    # print(canvas)
    k= cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
