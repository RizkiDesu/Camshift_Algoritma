import numpy as np
import cv2 as cv
import pyfirmata
import csv
# namafile = input("nama percobaan: ")
def IOU(box1, box2):
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union 
	return iou

# inisialisasi kamera
cap = cv.VideoCapture(1)
if cap.isOpened()==False:
    print("camera not accesss !!!")
    exit()
    
# memilih frame terbaik
count = 0
while(1):
    ret, frame = cap.read()
    cv.imshow('camshift', frame)
    cv.imwrite("frame_1/frame%d.jpg" % count, frame)
    count = count + 1
    if cv.waitKey(1) & 0xFF == ord('c'):
        print("terpilih frame ke:",count)
        count = 0
        break
    
# data train
crop = cv.selectROI("camshift", frame, False)
x_crop = int(crop[0])
y_crop = int(crop[1])
w_crop = int(crop[2])
h_crop = int(crop[3])
# cv.imwrite("frame_2/koordinat_data_train: %d, %r, %r, %r.jpg" %(x_crop, y_crop, w_crop, h_crop),)
# cv.imwrite("frame_2/model: %d, %r, %r, %r.jpg" %(x_crop, y_crop, w_crop, h_crop), frame)
cv.imwrite("frame_2/frame ;%d;%d;%r;%r.jpg" %(x_crop, y_crop, w_crop, h_crop), frame)
print("koordinat_data_train: %d, %r, %r, %r" %(x_crop, y_crop, w_crop, h_crop))
print("------------------------------------------")

# area
pembatas = cv.selectROI("camshift", frame, False)
x_pembatas = int(pembatas[0])
y_pembatas = int(pembatas[1])
w_pembatas = int(pembatas[2])
h_pembatas = int(pembatas[3])
# cv.imwrite("frame_2/frame;%d;%d;%r;%r.jpg" %(x_crop, y_crop, w_crop, h_crop), frame)
print("koordinat_pembatas: %d, %r, %r, %r" %(x_pembatas, y_pembatas, w_pembatas, h_pembatas))
print("------------------------------------------")

track_window = (x_crop, y_crop, w_crop, h_crop)
roi = frame[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]

cv.imwrite("roi/roi_rgb.jpg", roi)

#convert  roi to hsv
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

cv.imwrite("roi/roi_hsv.jpg", hsv_roi)

#menentukan mask
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))

cv.imwrite("roi/roi_mask.jpg", mask)


# calculate histogram a value
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])


# normalize these value
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)


# normalize these value
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)



# pengturan pyfirmata
port = "COM3"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:8:s') #pin 8 Arduino
servoX = 90

#inisiasi video
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('record/output.mp4',fourcc, 30.0, (640,480))

#tulis csv
file_csv = 'analisa/model %d %d %r %r.csv' %(x_crop, y_crop, w_crop, h_crop)
with open(file_csv, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["frame", "x_prediksi", "y_prediksi", "w_prediksi", "h_prediksi", "nilai_IOU", "nilai_servoX"])
    
    
    while (1):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 90)
            out.write(frame)
            
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            ret, track_window = cv.CamShift(dst, track_window, term_crit)
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            # final_image = cv.polylines(frame, [pts], True, (0, 255, 0), 3)
            x_prediksi = int(min(pts[:, 0]))
            y_prediksi = int(min(pts[:, 1]))
            w_prediksi = int(max(pts[:, 0])) - x_prediksi
            h_prediksi = int(max(pts[:, 1])) - y_prediksi
            
            final_image = cv.rectangle(frame, (x_prediksi, y_prediksi), (x_prediksi + w_prediksi, y_prediksi + h_prediksi), (255, 0, 0), 3)
            final_image = cv.rectangle(frame, (x_pembatas, y_pembatas), (x_pembatas + w_pembatas, y_pembatas + h_pembatas), (255, 255, 0), 3)
            
            cv.imshow('camshift', final_image)
            cv.imwrite("frame_3/frame_%d.jpg" %count, frame)
            akurasi = IOU([x_pembatas, y_pembatas, x_pembatas+w_pembatas, y_pembatas+h_pembatas], [x_prediksi, y_prediksi, x_prediksi+w_prediksi, y_prediksi+h_prediksi])
            
            writer.writerow([count, x_prediksi, y_prediksi, w_prediksi, h_prediksi, akurasi, servoX])
            if x_pembatas > x_prediksi < x_pembatas+w_pembatas or x_pembatas > x_prediksi+w_prediksi < x_pembatas+w_pembatas:
                if servoX == 0:
                    servoX -= 0
                else:
                    servoX -=1
                
            elif x_pembatas < x_prediksi > x_pembatas+w_pembatas or x_pembatas < x_prediksi+w_prediksi > x_pembatas+w_pembatas:
                if servoX == 180:
                    servoX +=0
                else:
                    servoX +=1
            servo_pinX.write(servoX)
            count += 1
            key = cv.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
print("file csv : ", file_csv)
cv.destroyAllWindows()
