# coding=UTF-8
import cv2
import os
import argparse
import time
import numpy as np
from os import listdir
from os.path import isfile, join

currentClassID = 0
tmpPoints = []
tmpLabels = np.empty(0)

g_image_size = (720, 720)
g_label_fmt = '%d', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f', '%5.4f'
g_print_help = 1

className = [
    "Fire_hydrant_box", 
    "Safety_exit_signs", 
    "Door", 
    "Crossroads", 
    "T-Junction"
]


def parse_args():
    parser = argparse.ArgumentParser(description='labeler')
    parser.add_argument('--original_image_dir', default='./original_images',
                        type=str, help='Original image storage path')
    parser.add_argument('--image_dir', default='./images',
                        type=str, help='Labeled image storage path')
    parser.add_argument('--label_dir', default='./labels',
                        type=str, help='Generated label storage path')
    parser.add_argument('--skip', default=False, action='store_true', 
                        help='Skip the labeled image')
    args = parser.parse_args()
    return args


def xywh2xy(box, flag):
    if flag == 0:
        return (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2))
    elif flag == 1:
        return (int(box[0] + box[2] / 2), int(box[1] - box[3] / 2))
    elif flag == 2:
        return (int(box[0] - box[2] / 2), int(box[1] + box[3] / 2))
    elif flag == 4:
        return (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
    

def drow_box_with_4points(image, label, fontScale = 1.1):
    reduction_label = label.copy()
    reduction_label[1:13:2] = reduction_label[1:13:2] * image.shape[1]
    reduction_label[2:13:2] = reduction_label[2:13:2] * image.shape[0]
    
    box = reduction_label[1:5]
    cv2.rectangle(image, xywh2xy(box, 0), xywh2xy(box, 4), (255, 255, 0), thickness=1)
    for i in range(5,13,2):
        x = reduction_label[i]
        y = reduction_label[i+1]
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)
    cv2.putText(image, '{}'.format(className[int(reduction_label[0])]), xywh2xy(box, 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (255, 0, 0), thickness=1)


def drawing(image, image_path, label_path):
    global tmpLabels, tmpPoints
    fontScale = 1
    image_show = image.copy()
    if g_print_help:
        p = image_show.shape[0] - 3
        line_h = 22 #pixel
        for i in range(len(className)):
            if i == currentClassID:
                cv2.putText(image_show, '{}:{}'.format(i, className[i]), (0, p - i * line_h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1)
            else:
                cv2.putText(image_show, '{}:{}'.format(i, className[i]), (0, p - i * line_h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=1)
        text = "[h]:print this help,[Esc]:exit,[0~9]:class,[ ]:save,[w]:next,[q]:previous,[c]:clear,[b]:undo,[Del/d]:del"
        words = text.split(',')
        for i in range(len(words)):
            cv2.putText(image_show, words[i], (0, (i + 1) * line_h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=1)
    else:
        cv2.putText(image_show, '[-h] for help', (0, image_show.shape[0] - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=1)

    if isfile(label_path):
        tmpLabels = np.loadtxt(label_path).reshape(-1, 13)
        # label : class x y w h x1 y1 x2 y2 x3 y3 x4 y4
        for i in range(tmpLabels.shape[0]):
            drow_box_with_4points(image_show, tmpLabels[i, ...], fontScale)
    else:
        if tmpLabels.size:
            for i in range(tmpLabels.shape[0]):
                drow_box_with_4points(image_show, tmpLabels[i, ...], fontScale)
        i = 0
        while i < len(tmpPoints):
            x = tmpPoints[i]
            y = tmpPoints[i+1]
            cv2.circle(image_show, (int(x), int(y)), 3, (255, 0, 255), thickness=-1)
            cv2.putText(image_show, 'No.{}:({},{})'.format(i // 2, x, y), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (255, 0, 0), thickness=1)
            i += 2
    cv2.imshow("image", image_show)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global currentClassID, tmpLabels, tmpPoints
    if event == cv2.EVENT_LBUTTONDOWN:
        tmpPoints.append(x)
        tmpPoints.append(y)
        if 8 == len(tmpPoints):
            tmp = []
            tmp.append(currentClassID)
            xs = [x / param[0].shape[1] for x in tmpPoints[0:7:2]]
            ys = [x / param[0].shape[0] for x in tmpPoints[1:8:2]]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            w = maxx - minx
            h = maxy - miny
            x = (maxx + minx) / 2
            y = (maxy + miny) / 2
            tmp.append(x)
            tmp.append(y)
            tmp.append(w)
            tmp.append(h)
            for i in range(4):
                tmp.append(xs[i])
                tmp.append(ys[i])

            tmp = np.array(tmp).reshape(-1, 13)
            if 0 == tmpLabels.size:
                tmpLabels = tmp
            else:
                tmpLabels = np.vstack((tmpLabels, tmp))
            tmpPoints = []
        drawing(param[0], param[1], param[2])

def soft_resize(image, size):
    h, w, _ = image.shape
    scale_h = size[0] / h
    scale_w = size[1] / w
    scale = min(scale_h, scale_w)
    if scale < 1:
        resize = (int(w * scale + 0.5), int(h * scale + 0.5))
        return cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    else:
        return image


def main(args):
    global currentClassID, tmpLabels, tmpPoints, g_image_size, g_print_help
    if args.skip:
        original_images = [f for f in listdir(args.original_image_dir) if isfile(join(args.original_image_dir, f))]
        labeled_images = [f for f in listdir(args.image_dir) if isfile(join(args.image_dir, f))]
        # labeled_images_without_suffix = [f.split('.')[0] for f in labeled_images]
        # image_files = [f for f in original_images if f.split('.')[0] not in labeled_images_without_suffix]
        image_files = list(set(original_images).difference(set(labeled_images)))
    else:
        image_files = [f for f in listdir(args.original_image_dir) if isfile(join(args.original_image_dir, f))]

    cv2.namedWindow("image")
    imgIdx = 0
    image_count = 0
    image_num = len(image_files)
    label_duration = 0
    while len(image_files) > 0 and imgIdx >= 0 and imgIdx < len(image_files):
        tik = time.time()
        # init
        tmpPoints = []
        tmpLabels = np.empty(0)
        
        image_name = image_files[imgIdx]
        original_image_path = join(args.original_image_dir, image_name)
        # labeled_image_path = join(args.image_dir, image_name.replace(image_name.split('.')[-1], 'jpg'))
        labeled_image_path = join(args.image_dir, image_name)
        label_path = join(args.label_dir, image_name).replace(image_name.split(".")[-1], "txt")

        image = cv2.imread(original_image_path)
        if image is None:
            print(image_name, "is Error!")
            os.remove(original_image_path)
            image_files.remove(image_name)
            continue
        
        resized_image = soft_resize(image, g_image_size)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[resized_image, labeled_image_path, label_path])
        drawing(resized_image, labeled_image_path, label_path)
        while True:
            key = cv2.waitKey()    # "[h]:help [Esc]:exit [0~9]:class [ ]:save [w]:next [q]:previous [c]:clear [b]:undo [Del d]:delete"
            # if -1 != key:
            #     print(key)
            
            if 27 == key: # 'Esc'
                return
            
            if 48 <= key and len(className) + 48 > key: # 0~9
                currentClassID = key - 48
                drawing(resized_image, labeled_image_path, label_path)

            if 32 == key: # /space
                cv2.imwrite(labeled_image_path, resized_image)
                np.savetxt(label_path, tmpLabels, fmt = g_label_fmt)
                imgIdx += 1
                break

            if 119 == key: # next
                imgIdx += 1
                if imgIdx >= len(image_files):
                    imgIdx = 0
                    print("Has returned to the head.")
                break

            if 113 == key: # previous
                imgIdx -= 1
                if imgIdx < 0:
                    imgIdx = len(image_files) - 1
                    print("Has returned to the tail.")
                break

            if 99 == key: # clear
                tmpLabels = np.empty(0)
                tmpPoints = []
                if isfile(label_path):
                    os.remove(label_path)
                if isfile(labeled_image_path):
                    os.remove(labeled_image_path)
                drawing(resized_image, labeled_image_path, label_path)
                break

            if 98 == key: # back
                if len(tmpPoints) >= 2:
                    tmpPoints = tmpPoints[:-2]
                elif tmpLabels.size:
                    tmpPoints =  tmpLabels[-1, 5:-2]
                    tmpLabels = tmpLabels[:-1, :]
                    tmpPoints[0:6:2] = (tmpPoints[0:6:2] * resized_image.shape[1]) + 0.5
                    tmpPoints[1:6:2] = (tmpPoints[1:6:2] * resized_image.shape[0]) + 0.5
                    tmpPoints = tmpPoints.astype(np.int32).tolist()
                drawing(resized_image, labeled_image_path, label_path)

            if 255 == key or 100 == key: # del image
                os.remove(original_image_path)
                if isfile(labeled_image_path):
                    os.remove(labeled_image_path)
                if isfile(label_path):
                    os.remove(label_path)
                image_files.remove(image_name)
                print("--Removed", original_image_path)
                break
            
            if 104 == key: # print help ?
                g_print_help = (g_print_help + 1) % 2
                drawing(resized_image, labeled_image_path, label_path)
        tok = time.time()
        duration = tok - tik
        if imgIdx == 0:
            label_duration = duration
            image_count += 1
        elif duration <= 15:
            label_duration += duration
            image_count += 1
        print('当前为第 {} 张图片，已用时 {:.2f} 分钟'.format(imgIdx+1, label_duration/60))
        print('剩余 {} 张图片，预估剩余时间 {:.2f} 分钟\n'.format(
            image_num-imgIdx - 1, label_duration/60/image_count*(image_num-image_count)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    