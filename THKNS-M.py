import numpy as np
import cv2
import pandas as pd
import pylab as plt
import skimage.io
from skimage.feature import canny
import math as ma
import time
from collections import Counter
import os
import xlsxwriter


def measurment(OsPath,ImgPath,FinalPath):
    '''

    :param OsPath: Diretory in a windows operating system. It's only purpose is to mave back to it's orginal directory
    :param ImgPath: Directory path to the image in a windows operating system
    :param FinalPath: Final Directory path
    :return:
    '''
    name = ImgPath.split('/')[-1] ## Grabing the name of the image: Sample.jpg
    name = name.split('.')[0]## Grabing the name of the file: Sample

    img = skimage.io.imread(ImgPath) ## Reading in image
    skimage.io.imshow(img)## Displaying image

    plt.title(f"{name}: Pre processing sample image")
    plt.show()

    final_img = img.copy()
    scale_img = img.copy()

    ###
    ## The following nested loop is made to find the scale bar.
    ## In this case the scale bar is always blue
    ## This can ofcouse be changed if the scale bar is green or red
    ###
    scale_loc = []
    scale_loc_x_y = []

    for row in range(scale_img.shape[0]):
        for col in range(scale_img.shape[1]):
            ## looking for the scale line thing
            if (scale_img[row, col][0] >= 0 and scale_img[row, col][0] <= 25) and \
                    (scale_img[row, col][1] >= 0 and scale_img[row, col][1] <= 25) and \
                    (scale_img[row, col][2] >= 200 and scale_img[row, col][2] <= 255):
                # print(f"Location of the scale line ish: (x,y): {col, row}")
                # scale_loc_x_y.append(col) #x
                # scale_loc_x_y.append(row) #y
                scale_loc.append([col, row])
            else:
                scale_img[row, col] = [0, 0, 0]

    scale_loc = np.array(scale_loc).reshape(len(scale_loc), 2)  ## [x,y] ## How the array will look like

    ## We're wanting to look for continuous values of scale
    ## Basically the most freq y value (row)
    ## This method below is find the total length of the scale bar and convert it to real dimentions
    # print(scale_loc[:,1])
    row_freq = []
    row_key = list(Counter(scale_loc[:, 1]).keys())
    row_val = list(Counter(scale_loc[:, 1]).values())
    row_freq = np.stack([row_key, row_val], axis=1)


    max_scale = np.where(row_freq[:, 1] == row_freq[:, 1].max())[0]
    if len(max_scale) > 1:
        max_scale = max_scale[0]

    max_row = row_freq[:, 0][max_scale]

    a = scale_loc[scale_loc[:, 1] == max_row]
    print("Length of scale bar: ", a[:, 0].min() - a[:, 0].max())
    dist_to_pix = 100 / abs(a[:, 0].min() - a[:, 0].max())  ## 100 micrometers / length of scale bar in px
    ## micrometers/px
    print(f"Pixel to distance ratio: {dist_to_pix} micrometers/px")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## Converting image to gray scale
    skimage.io.imshow(img_gray)
    plt.title(f"{name}: GrayScale image")
    plt.show()

    img_g_b = cv2.medianBlur(img_gray, 9)  ## img_g_b gray and blurred

    skimage.io.imshow(img_g_b)
    plt.title(f"{name}: MedianBlur")
    plt.show()

    edges = cv2.Canny(img_g_b, 50, 50)
    skimage.io.imshow(edges)
    plt.title(f"{name}: Canny applied")
    plt.show()


    ##############################
    ### Applying the hough algothrim to rotate all edge lines 90 degress to find an interseting line
    lines_list = []

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=3, maxLineGap=5)

    for i in lines:
        x1, y1, x2, y2 = i[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        Line1 = [x1, y1]
        Line2 = [x2, y2]
        lines_list.append([Line1, Line2])

    ## Lines_list come in pairs
    Lines_new = []
    for i in range(len(lines_list)):
        p1 = lines_list[i][0]
        p2 = lines_list[i][1]

        ## Find the mid point of these two lines
        Mx = (p1[0] + p2[0]) // 2  ## using // just so it's an int

        My = (p1[1] + p2[1]) // 2
        M = [Mx, My]

        ## normalize p1 and p2 to the midpoint (i.e: changing the reference frame)
        p1_x = p1[0] - Mx
        p1_y = p1[1] - My

        ## Apply the rotation matrix
        ##
        ## [0 -1][p1_x ---> -p1_y + Mx ---> New X location
        ## [1 0]  P1_y] --> p1_x + my ----> New Y location
        ##

        p1_x_new = -p1_y + Mx  ## New X
        p1_y_new = p1_x + My  ## New Y
        p1_new = [p1_x_new, p1_y_new]
        # print(f"New P1: {p1_new}")

        ##------------------------
        ##------ Redo for p2 -----
        ##________________________

        p2_x = p2[0] - Mx
        p2_y = p2[1] - My

        p2_x_new = -p2_y + Mx  ## New X
        p2_y_new = p2_x + My  ## New Y
        p2_new = [p2_x_new, p2_y_new]
        ## ----- Rotation matrix applied ----
        ## -- The line is now rotated by 90, and it now normal to the oringal line
        ### ----- If new line is less than 5 in magnitude, increase the length by 15
        del_xNew = p1_x_new - p2_x_new
        del_yNew = p1_y_new - p2_y_new
        mag_newLine = [del_xNew, del_yNew]
        length = np.hypot(mag_newLine[0], mag_newLine[1])
        delta_x = del_xNew / length
        delta_y = del_yNew / length
        c = 35  ## scalar to increase the length of the new line
        if length < 10:
            if np.hypot(p1_new[0], p1_new[1]) < np.hypot(p2_new[0], p2_new[1]):
                p1_new[0] -= int(c * delta_x)
                p1_new[1] -= int(c * delta_y)

                p2_new[0] += int(c * delta_x)
                p2_new[1] += int(c * delta_y)
                ### Draw line ###
                Lines_new.append([p1_new, p2_new])
                cv2.line(img, (p1_new[0], p1_new[1]), (p2_new[0], p2_new[1]), (0, 0, 255), 1)
            else:
                p1_new[0] += int(c * delta_x)
                p1_new[1] += int(c * delta_y)

                p2_new[0] -= int(c * delta_x)
                p2_new[1] -= int(c * delta_y)
                ### Draw line ###
                Lines_new.append([p1_new, p2_new])
                cv2.line(final_img, (p1_new[0], p1_new[1]), (p2_new[0], p2_new[1]), (0, 0, 255), 1)
        else:
            ### Draw line ###
            Lines_new.append([p1_new, p2_new])
            cv2.line(final_img, (p1_new[0], p1_new[1]), (p2_new[0], p2_new[1]), (0, 0, 255), 1)

    def get_intersection(L1, L2):
        '''

        :param L1: New line
        :param L2: Old line
        :return:
        '''
        int_point = []
        x1 = L1[0][0]
        y1 = L1[0][1]

        x2 = L1[1][0]
        y2 = L1[1][1]

        x3 = L2[0][0]
        y3 = L2[0][1]

        x4 = L2[1][0]
        y4 = L2[1][1]

        demon = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) ## Demoninator
        numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if demon == 0:
            return None
        t = numerator / demon


        if t > 1 or t < 0:
            return None

        if not ma.isnan(t):
            px = int(x1 + t * (x2 - x1))

            py = int(y1 + t * (y2 - y1))

        return [px, py]

    Point_int = ''  ## Point of intersection
    Thickness = []
    THCKS_PosData = []
    for i in Lines_new:
        pos = []

        for j in lines_list:

            CurrPoint = get_intersection(i, j)  ## Current Point
            ## Looping through all lines for all possible interestions.
            ## Though a lot of them wont have intersections.
            ## I couldnt find a way to make it anymore efficent
            T = [0, 0, 0]
            TL = [0, 0, 0]
            L = [0, 0, 0]
            BL = [0, 0, 0]
            B = [0, 0, 0]
            BR = [0, 0, 0]
            R = [0, 0, 0]
            TR = [0, 0, 0]

            if CurrPoint != None:

                if (0 <= CurrPoint[0] < img.shape[1]) and (0 <= (CurrPoint[1] - 1) < img.shape[0]):
                    ## If it's in range
                    T = img[CurrPoint[1] - 1, CurrPoint[0]]

                if (0 <= (CurrPoint[0] - 1) < img.shape[1]) and (0 <= (CurrPoint[1] - 1) < img.shape[0]):
                    TL = img[CurrPoint[1] - 1, CurrPoint[0] - 1]

                if (0 <= (CurrPoint[0] - 1) < img.shape[1]) and (0 <= (CurrPoint[1]) < img.shape[0]):
                    L = img[CurrPoint[1], CurrPoint[0] - 1]

                if (0 <= (CurrPoint[0] - 1) < img.shape[1]) and (0 <= (CurrPoint[1] + 1) < img.shape[0]):
                    BL = img[CurrPoint[1] + 1, CurrPoint[0] - 1]

                if (0 <= (CurrPoint[0]) < img.shape[1]) and (0 <= (CurrPoint[1] + 1) < img.shape[0]):
                    B = img[CurrPoint[1] + 1, CurrPoint[0]]

                if (0 <= (CurrPoint[0] + 1) < img.shape[1]) and (0 <= (CurrPoint[1] + 1) < img.shape[0]):
                    BR = img[CurrPoint[1] + 1, CurrPoint[0] + 1]

                if (0 <= (CurrPoint[0] + 1) < img.shape[1]) and (0 <= (CurrPoint[1]) < img.shape[0]):
                    R = img[CurrPoint[1], CurrPoint[0] + 1]

                if (0 <= (CurrPoint[0] + 1) < img.shape[1]) and (0 <= (CurrPoint[1] - 1) < img.shape[0]):
                    TR = img[CurrPoint[1] - 1, CurrPoint[0] + 1]

                if T[0] == 255 and T[1] == 0 and T[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if TL[0] == 255 and TL[1] == 0 and TL[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if L[0] == 255 and L[1] == 0 and L[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if BL[0] == 255 and BL[1] == 0 and BL[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if B[0] == 255 and B[1] == 0 and B[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if BR[0] == 255 and BR[1] == 0 and BR[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if R[0] == 255 and R[1] == 0 and R[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
                if TR[0] == 255 and TR[1] == 0 and TR[2] == 0:
                    pos.append(CurrPoint)  ## Point of intersection

                    continue
        # print(pos)

        if len(pos) == 2:
            ## record the positions and draw it, but also find the thickness
            pass

        else:
            max_dist = 0
            max_pos = None
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dist = np.hypot((pos[i][0] - pos[j][0]), (pos[i][1] - pos[j][1])) ## finding the hypotenus to find the real thickness value
                    if dist > max_dist:
                        max_dist = dist

                        max_pos = [pos[i], pos[j]]
                        ## This just find the maxium distance of a thickness measurment.
                        ## It gets stored and filtered out if it's in the bounds of the scale bar.

            ## Blocking out the area of the scale bar. Therefore it wont find any intersecting points in this region
            if max_pos != None and max_dist > 8 and (max_pos[0][0] <= 1390 and max_pos[0][1] <= 1110) and (
                    max_pos[1][0] <= 1390 and max_pos[1][1] <= 1110):
                Thickness.append(max_dist * dist_to_pix)
                THCKS_PosData.append([max_dist * dist_to_pix, np.array(max_pos)])

    THCKS_PosData = np.array(THCKS_PosData)
    DATA_std = np.std(THCKS_PosData[:,0])
    DATA_mean = np.mean(THCKS_PosData[:,0])


    ### Area of inprovement. A robust method of filtering out outliers is needed here.
    ZScore = abs(THCKS_PosData[:,0] - DATA_mean)/DATA_std
    print(f"Inner quantile: {np.quantile(THCKS_PosData[:,0], .75)}")
    Q3 = np.quantile(THCKS_PosData[:,0], .75)
    Q1 = np.quantile(THCKS_PosData[:, 0], .25)
    THCKS_PosData = np.column_stack((THCKS_PosData, ZScore))
    THCKS_PosData = THCKS_PosData[THCKS_PosData[:,2] < Q3]
    THCKS_PosData = THCKS_PosData[THCKS_PosData[:, 0] > Q1]
    postions = THCKS_PosData[:,1].tolist()
    for i,j in zip(postions, THCKS_PosData[:,0]):
        max_pos = i
        DIST = j
        cv2.line(final_img, (max_pos[0][0], max_pos[0][1]), (max_pos[1][0], max_pos[1][1]), (0, 255, 0), 1)
        print(f"max_dist in micrometers: {(DIST)}um  at pos: {max_pos}")
    Avg_tHKNS = np.mean(THCKS_PosData[:,0])
    THKNSSTD = np.std(THCKS_PosData[:,0])
    Q_25 = np.quantile(THCKS_PosData[:,0], .25)
    Q_925 = np.quantile(THCKS_PosData[:,0], .925)
    Q_50 = np.quantile(THCKS_PosData[:,0], .5)
    ### Look for outilers with Z-Score
    ### Cutoff: 3 STD

    print(f"Average: {np.mean(THCKS_PosData[:,0])} um\n"
          f"2.5% and 92.5% Qunatile [Low,Hi]: {np.quantile(THCKS_PosData[:,0], .25), np.quantile(THCKS_PosData[:,0], .925)} um\n"
          f"50th quantile: {np.quantile(THCKS_PosData[:,0], .5)} um, STD: {THKNSSTD}")

    os.chdir(FinalPath)
    cv2.imwrite(f'{name}Final.jpg', final_img)
    os.chdir(OsPath)
    skimage.io.imshow(final_img)
    plt.title(f"{name}: Final image")
    plt.show()
    df = pd.DataFrame()

    writer = pd.ExcelWriter(f"{name} Measurments.xlsx")

    df["Thickness"] = Thickness
    df["Average"] = Avg_tHKNS
    df["2.5% qunatile"] = Q_25
    df["50% qunatile"] = Q_50
    df["92.5% quantile"] = Q_925
    df.to_excel(writer, sheet_name='Sheet1')

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.insert_image('G3', f'{name}Final.jpg')

    writer.close()
    return writer


def main (FilePath: str) -> str:
    '''
    :param FilePath: String: Path that contains all the images
    :return: Returns the thickness measurments
    '''
    path = FilePath.replace(os.sep, '/')
    os.chdir(FilePath)

    FinalImg_Directory = "Final_Images"
    if not os.path.exists(FinalImg_Directory):
        # If it doesn't exist, create it
        os.makedirs(FinalImg_Directory)
    print(path)

    for imgs in os.listdir(path):
        ## Check for all jpeg, jpg, png
        if (imgs.endswith(".png") or imgs.endswith(".jpg") or imgs.endswith(".jpeg")):
            imgpath = path + '/' + imgs

            measurment(FilePath, imgpath,FinalImg_Directory)


    return 0

if __name__ == "__main__":
    path = input(r'File Path:   ')
    start = time.time()
    print(main(path))
    end = time.time()
    print(f"Time of execution: {end - start}")