import cv2
import os
import glob

data_dir = sorted(glob.glob('dataset/dashcam/testing/positive/*.mp4'))[0:20]
for i in range(len(data_dir)):
    if i < 10:
        os.mkdir('dataset/dashcam/tmp_testing/0' + str(i))
    else:
        os.mkdir('dataset/dashcam/tmp_testing/' + str(i))
for i in range(len(data_dir)):
    cam = cv2.VideoCapture(data_dir[i])
    currentframe = 0
    while (True):
        ret, frame = cam.read()
        if ret:
            if i < 10:
                if currentframe < 10:
                    name = 'dataset/dashcam/tmp_testing/0' + str(i) + '/00' + str(currentframe) + '.jpg'
                elif currentframe >= 10 and currentframe <= 99:
                    name = 'dataset/dashcam/tmp_testing/0' + str(i) + '/0' + str(currentframe) + '.jpg'
                else:
                    name = 'dataset/dashcam/tmp_testing/0' + str(i) + '/' + str(currentframe) + '.jpg'
            else:
                if currentframe < 10:
                    name = 'dataset/dashcam/tmp_testing/' + str(i) + '/00' + str(currentframe) + '.jpg'
                elif currentframe >= 10 and currentframe <= 99:
                    name = 'dataset/dashcam/tmp_testing/' + str(i) + '/0' + str(currentframe) + '.jpg'
                else:
                    name = 'dataset/dashcam/tmp_testing/' + str(i) + '/' + str(currentframe) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()