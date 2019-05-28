import numpy as np
import cv2
import os


def compute_fit_adjust(array, arrayRef):
    """
    This functions computes the 3D Fit Adjust and returns the computed value
    according to the formula:

    .. math::
    Fit = 1 - (A_ref ∩ A_seg) / (A_ref ∪ A_seg).

    Arguments:
        array {numpy.array} -- input 3D array with segmented content
        array {numpy.array} -- input 3D array with reference content

    Returns:
        float -- Fit Adjust value between 0 and 1.
    """
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = np.sum(imand)
    sumor = np.sum(imor)

    result = (sumand / float(sumor))

    return result


def compute_size_adjust(array, arrayRef):
    """
    This functions computes the 2D Size Adjust and returns the computed value:

    Arguments:
        array {numpy.array} -- input 2D array with segmented content
        array {numpy.array} -- input 2D array with reference content

    Returns:
        float -- Fit Adjust value between 0 and 1.
    """
    imArea1 = np.count_nonzero(arrayRef)
    imArea2 = np.count_nonzero(array)
    subArea = np.abs(imArea1 - imArea2)
    sumArea = imArea1 + imArea2

    result = (1 - subArea / sumArea)

    return result


def compute_position_adjust(arraySeg, arrayRef):
    """
    This functions computes the 2D Size Adjust and returns the computed value:

    Arguments:
        array {numpy.array} -- input 2D array with segmented content
        array {numpy.array} -- input 2D array with reference content

    Returns:
        float -- Fit Adjust value between 0 and 1.
    """
    indsSeg = np.where(arraySeg > 0)
    indsRef = np.where(arrayRef > 0)

    centroidRefY = indsRef[0].mean()
    centroidRefX = indsRef[1].mean()

    centroidSegY = indsSeg[0].mean()
    centroidSegX = indsSeg[1].mean()

    subCentroidY = np.abs(centroidSegY - centroidRefY) / arrayRef.shape[0]
    subCentroidX = np.abs(centroidSegX - centroidRefX) / arrayRef.shape[1]

    result = 1 - (subCentroidY + subCentroidX) / 3

    return result


def compute_dice_similarity(array, arrayRef):
    """
    This functions computes the 2D Dice Similarity Coefficient and returns the computed value
    according to the formula:

    .. math::
    DSC = 2 * (A_seg ∩ A_ref) (|A_seg| + |A_ref|)

    Arguments:
        array {numpy.array} -- input 2D array with segmented content
        array {numpy.array} -- input 2D array with reference content

    Returns:
        float -- Fit Adjust value between 0 and 1.
    """
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = 2 * np.sum(imand)
    sumor = np.sum(array) + np.sum(arrayRef)

    result = (sumand / float(sumor))

    return result



path_seg = "C:/googleDrive/images_result_adalberto/seg/"
path_gold = "C:/googleDrive/images_result_adalberto/gold/"
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


res_dice = compute_dice_similarity(lst_seg[0], lst_gold[0])
res_fit = compute_fit_adjust(lst_seg[0], lst_gold[0])
res_size = compute_size_adjust(lst_seg[0], lst_gold[0])
res_pos = compute_position_adjust(lst_seg[0], lst_gold[0])


print("Dice Similarity: {}".format(res_dice))
print("Fitness Adjust:  {}".format(res_fit))
print("Size Adjust:     {}".format(res_size))
print("Position Adjust: {}".format(res_pos))