import cv2
import os
import numpy as np
from imageio import imread, imsave

def getBinaryImage(img, flag):

    if flag == True:
        # Сглаживание и бинаризация всего изображения
        img_blur = cv2.bilateralFilter(img, 30, 90, 90)
        cv2.imwrite('output/Blur1.jpg', img_blur)
        img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 3)
        cv2.imwrite('output/Bin1.jpg', img_bin)
    else:
        # Сглаживание, бинаризация и морфологическое замыкание части изображения с предметами
        img_blur = cv2.medianBlur(img, 21, 81)
        cv2.imwrite('output/Blur2.jpg', img_blur)
        img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('output/Bin2.jpg', img_bin)

    return img_bin

# Получение контуров на изображении
def getContours(img_bin, img_orig, flag):

    if flag == True:
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # находим контуры
    else:
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    temp = []
    for cont in contours:
        # находим площадь контуров
        area = cv2.contourArea(cont)
        if area > 10:
            temp.append([cont, area])

    # сортируем контуры по площади
    area_contours_sorted = sorted(temp, key=lambda cont: cont[1])

    return area_contours_sorted

# Получение вершин многоугольника и листа бумаги
def getVertex(contour, img_orig, flag):

    font = cv2.FONT_HERSHEY_COMPLEX

    vertex = []

    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
    n = approx.ravel()

    i = 0
    for j in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]

            if flag == True:
                string = "(" + str(x) + "," + str(y) + ")"
                cv2.putText(img_orig, string, (x, y), font, 2, (0,0 ,255), 3)

            vertex.append((x))

        i = i + 1

    if flag == True:
        cv2.imwrite('output/Vertex.jpg', img_orig)

    return vertex

# Получение границы белого листа и разделение изображения на два: с предметами и с листом
def getBorderAndCut(vertex, img_orig, img_bin):

    border_x = min(vertex)

    cv2.line(img_orig, (border_x, 0), (border_x, 3000), (255, 0, 30), 10, cv2.LINE_AA)
    cv2.imwrite('output/Border.jpg', img_orig)

    img_objects = img_bin[0:img_bin.shape[1], 0:border_x-30]
    img_paper = img_bin[0:img_bin.shape[1], border_x:img_bin.shape[0]]
    cv2.imwrite('output/Objects.jpg', img_objects)

    return img_objects, img_paper

# Получение ограничивающего прямогугольника предметов
def getBoundingBox(contours, img_orig):

    box = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt[0])
        if h > 100:
            box.append([x, y, w, h])
            img = cv2.rectangle(img_orig, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imwrite('output/Contours.jpg', img)

    return box


def main():

    path = 'input/2.jpg'

    if not os.path.exists(path):
        return False
    img_orig = cv2.imread(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Проверка разрешения изображения
    if img.shape[0] * img.shape[1] != 4000 * 3000:
        return False

    # Сглаживание и бинаризация исходного изображения
    img_bin = getBinaryImage(img, True)

    if img_bin is None:
        return False

    # Получение контуров на изображении
    contours = getContours(img_bin, img_orig, True)

    if contours == None:
        return False
    else:
        # Вывод контура белого листа
        img_cont = cv2.drawContours(img_orig, contours[-1][0], -1, (255, 0, 255), 5)
        cv2.imwrite('output/Contours.jpg', img_cont)

    # Получение вершин белого листа
    paper_vertex = getVertex(contours[-1][0], img_orig, True)

    # Разделение изображения на два: с предметами и с листом
    img_objects, img_paper = getBorderAndCut(paper_vertex, img_orig, img)

    # Получение контуров изображения с листом
    contours = getContours(img_paper, img_paper, True)

    # Получение вершин многоугольника
    poly_vertex = getVertex(contours[-1][0], img_orig, False)
    # Проверка, что у многоугольника не больше 6 вершин включительно
    if len(poly_vertex) > 6:
        return False

    #  Сглаживание, бинаризация и морфологическое замыкание изображения с предметами
    img_bin = getBinaryImage(img_objects, False)

    # Нахождение контуров на изображении с предметами
    contours = getContours(img_bin, img_objects, False)

    # Получение ограничивающего прямогугольника предметов
    box = getBoundingBox(contours, img_objects)

    return True


if __name__ == '__main__':
    print(main())



