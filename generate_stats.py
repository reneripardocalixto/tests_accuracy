import cv2
import os

path_seg = "data/Seg/"
path_gold = "data/Gold/"
seg_images = os.listdir(path_seg)
gold_images = os.listdir(path_gold)

lst_seg = []
lst_gold = []


for f in seg_images:
    im = cv2.imread(path_seg + f, cv2.IMREAD_GRAYSCALE)
    lst_seg.append(im)

    # cv2.imshow('teste', im)
    # cv2.waitKey()


for f in gold_images:
    im = cv2.imread(path_gold + f, cv2.IMREAD_GRAYSCALE)
    lst_gold.append(im)

    # cv2.imshow('teste', im)
    # cv2.waitKey()


def compute_detection_result(lst_gold, lst_seg):
    count_detected = 0
    count_fp = 0
    count_all_contours = 0
    total_images = len(lst_gold)
    for i, img in enumerate(lst_seg):
        img_gold = lst_gold[i]
        print(img_gold.shape)

        img_detected = cv2.bitwise_and(img, img_gold)
        img_fp = cv2.bitwise_and(img, cv2.bitwise_not(img_gold))

        # cv2.imshow('intersection', cv2.resize(img_fp, (640,640)) )
        # cv2.waitKey()

        _, cnt_img, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, cnt_gold, _ = cv2.findContours(img_gold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, cnt_fp, _ = cv2.findContours(img_fp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, cnt_detected, _ = cv2.findContours(img_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if img_detected.max() > 100:
            # print("{}: detected".format(i+1))
            count_detected += 1
        else:
            count_fp += len(cnt_fp)
            # print(i+1)

        count_all_contours += len(cnt_img) + len(cnt_gold)

    print("Total detected accuracy: {}".format(count_detected / total_images))
    print("Total False Positive: {}".format(count_fp / total_images))


compute_detection_result(lst_gold, lst_seg)