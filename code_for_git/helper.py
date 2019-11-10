import numpy as np
import keras
from numpy import argmax
import tensorflow as tf
import cv2
import random
import os
from numpy.random import RandomState

# create parallel random number to get same occlusion and
# corresponding segmentation mask on the fly

random_state_1 = RandomState(1234)
random_state_2 = RandomState(4321)

random_state_x1 = RandomState(123)
random_state_x2 = RandomState(123)
random_state_y1 = RandomState(321)
random_state_y2 = RandomState(321)

def ICC31(ground_truth, prediction):
    """
       Calculates Interclass Correlation Coefficient (3,1) as defined in
       P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in
       Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.
    """
    # remove NaN values
    idx = np.squeeze(~np.isnan(ground_truth))
    ground_truth = ground_truth[idx]
    prediction = prediction[idx]

    dat = np.column_stack((ground_truth, prediction))
    # number of raters/ratings
    k = 2;
    # number of targets
    n = dat.shape[0];
    # mean per target
    mpt = np.mean(dat, axis=1);
    mpt.shape = (n,1);
    # mean per rater/rating
    mpr = np.mean(dat, axis=0);
    # get total mean
    tm = np.mean(mpt);
    # within target sum sqrs
    WSS = np.sum(np.sum((dat-mpt)**2));
    # within target mean sqrs
    WMS = WSS / (n * (k - 1));
    # between rater sum sqrs
    RSS = np.sum((mpr - tm)**2) * n;
    # between rater mean sqrs
    RMS = RSS / (k - 1);
    # between target sum sqrs
    BSS = np.sum((mpt - tm)**2) * k;
    # between targets mean squares
    BMS = BSS / (n - 1);
    # residual sum of squares
    ESS = WSS - RSS;
    # residual mean sqrs
    EMS = ESS / ((k - 1) * (n - 1));
    # ICC(3,1)
    return (BMS - EMS) / (BMS + (k - 1) * EMS)

class Metrics(tf.keras.callbacks.Callback):

    """
    for regression
    take val data at every epoch end
    calculate ICC31 for every Action Units
    then append log for every AUs and mean
    """
    def on_epoch_end(self, batch, logs={}):
        predict_argmax = np.asarray(self.model.predict(self.validation_data[0]))
        targ_argmax = self.validation_data[1]

        targ1 = targ_argmax[:, 0]
        targ2 = targ_argmax[:, 1]
        targ3 = targ_argmax[:, 2]
        targ4 = targ_argmax[:, 3]
        targ5 = targ_argmax[:, 4]
        targ6 = targ_argmax[:, 5]
        targ7 = targ_argmax[:, 6]
        targ8 = targ_argmax[:, 7]
        targ9 = targ_argmax[:, 8]
        targ10 = targ_argmax[:, 9]
        targ11 = targ_argmax[:, 10]
        targ12 = targ_argmax[:, 11]
        targ13 = targ_argmax[:, 12]
        targ14 = targ_argmax[:, 13]
        targ15 = targ_argmax[:, 14]
        targ16 = targ_argmax[:, 15]
        targ17 = targ_argmax[:, 16]
        targ18 = targ_argmax[:, 17]
        targ19 = targ_argmax[:, 18]
        targ20 = targ_argmax[:, 19]
        targ21 = targ_argmax[:, 20]
        targ22 = targ_argmax[:, 21]
        targ23 = targ_argmax[:, 22]
        targ24 = targ_argmax[:, 23]
        targ25 = targ_argmax[:, 24]
        targ26 = targ_argmax[:, 25]

        pred_1 = predict_argmax[:, 0]
        pred_2 = predict_argmax[:, 1]
        pred_3 = predict_argmax[:, 2]
        pred_4 = predict_argmax[:, 3]
        pred_5 = predict_argmax[:, 4]
        pred_6 = predict_argmax[:, 5]
        pred_7 = predict_argmax[:, 6]
        pred_8 = predict_argmax[:, 7]
        pred_9 = predict_argmax[:, 8]
        pred_10 = predict_argmax[:, 9]
        pred_11 = predict_argmax[:, 10]
        pred_12 = predict_argmax[:, 11]
        pred_13 = predict_argmax[:, 12]
        pred_14 = predict_argmax[:, 13]
        pred_15 = predict_argmax[:, 14]
        pred_16 = predict_argmax[:, 15]
        pred_17 = predict_argmax[:, 16]
        pred_18 = predict_argmax[:, 17]
        pred_19 = predict_argmax[:, 18]
        pred_20 = predict_argmax[:, 19]
        pred_21 = predict_argmax[:, 20]
        pred_22 = predict_argmax[:, 21]
        pred_23 = predict_argmax[:, 22]
        pred_24 = predict_argmax[:, 23]
        pred_25 = predict_argmax[:, 24]
        pred_26 = predict_argmax[:, 25]

        ICC31_au1 = ICC31(targ1, pred_1)
        ICC31_au2 = ICC31(targ2, pred_2)
        ICC31_au3 = ICC31(targ3, pred_3)
        ICC31_au4 = ICC31(targ4, pred_4)
        ICC31_au5 = ICC31(targ5, pred_5)
        ICC31_au6 = ICC31(targ6, pred_6)
        ICC31_au7 = ICC31(targ7, pred_7)
        ICC31_au8 = ICC31(targ8, pred_8)
        ICC31_au9 = ICC31(targ9, pred_9)
        ICC31_au10 = ICC31(targ10, pred_10)
        ICC31_au11 = ICC31(targ11, pred_11)
        ICC31_au12 = ICC31(targ12, pred_12)
        ICC31_au13 = ICC31(targ13, pred_13)
        ICC31_au14 = ICC31(targ14, pred_14)
        ICC31_au15 = ICC31(targ15, pred_15)
        ICC31_au16 = ICC31(targ16, pred_16)
        ICC31_au17 = ICC31(targ17, pred_17)
        ICC31_au18 = ICC31(targ18, pred_18)
        ICC31_au19 = ICC31(targ19, pred_19)
        ICC31_au20 = ICC31(targ20, pred_20)
        ICC31_au21 = ICC31(targ21, pred_21)
        ICC31_au22 = ICC31(targ22, pred_22)
        ICC31_au23 = ICC31(targ23, pred_23)
        ICC31_au24 = ICC31(targ24, pred_24)
        ICC31_au25 = ICC31(targ25, pred_25)
        ICC31_au26 = ICC31(targ26, pred_26)
        ICC31_mean = ICC31(targ_argmax.flatten(), predict_argmax.flatten())

        logs.update({"ICC31_mean": ICC31_mean, "ICC31_au1": ICC31_au1, "ICC31_au2": ICC31_au2, "ICC31_au4": ICC31_au3,
                     "ICC31_au5": ICC31_au4, "ICC31_au6": ICC31_au5, "ICC31_au7": ICC31_au6, "ICC31_au9": ICC31_au7,
                     "ICC31_au10": ICC31_au8, "ICC31_au11": ICC31_au9, "ICC31_au12": ICC31_au10,
                     "ICC31_au14": ICC31_au11,
                     "ICC31_au15": ICC31_au12, "ICC31_au16": ICC31_au13, "ICC31_au17": ICC31_au14,
                     "ICC31_au18": ICC31_au15,
                     "ICC31_au20": ICC31_au16, "ICC31_au22": ICC31_au17, "ICC31_au23": ICC31_au18,
                     "ICC31_au24": ICC31_au19,
                     "ICC31_au25": ICC31_au20, "ICC31_au26": ICC31_au21, "ICC31_au27": ICC31_au22,
                     "ICC31_au28": ICC31_au23, "ICC31_au34": ICC31_au24, "ICC31_au38": ICC31_au25,
                     "ICC31_au43": ICC31_au26})
        print(ICC31_mean, 'ICC31_mean')
        return

def mean_(numbers):
    """
    to use with custom metrics with data generator
    """
    return float(sum(numbers)) / max(len(numbers), 1)

class Metrics_2(keras.callbacks.Callback):
    """
    custom metrics to calculate ICC31 with keras image data-generator
    """
    def __init__(self, val_data, batch_size=100):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        total = batches * self.batch_size
        avg = []
        for batch in range(batches):
            xVal, yVal = next(self.validation_data)
            pred = np.asarray(self.model.predict(xVal))
            targ = yVal
            ICC31_ = ICC31(argmax(targ, axis=2), argmax(pred, axis=2))
            avg.append(ICC31_)
        ICC31_mean = mean_(avg)
        logs.update({"ICC31_mean": ICC31_mean})
        print(ICC31_mean, "ICC31_mean")
        return

class Metrics_classification(keras.callbacks.Callback):
    """
    for classification
    take val data at every epoch end
    calculate ICC31 for every Action Units
    then append log for every AUs and mean
    """
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        predict_argmax = argmax(predict, axis=2)
        targ_argmax = argmax(targ, axis=2)

        targ1 = targ_argmax[:, 0]
        targ2 = targ_argmax[:, 1]
        targ3 = targ_argmax[:, 2]
        targ4 = targ_argmax[:, 3]
        targ5 = targ_argmax[:, 4]
        targ6 = targ_argmax[:, 5]
        targ7 = targ_argmax[:, 6]
        targ8 = targ_argmax[:, 7]
        targ9 = targ_argmax[:, 8]
        targ10 = targ_argmax[:, 9]
        targ11 = targ_argmax[:, 10]
        targ12 = targ_argmax[:, 11]
        targ13 = targ_argmax[:, 12]
        targ14 = targ_argmax[:, 13]
        targ15 = targ_argmax[:, 14]
        targ16 = targ_argmax[:, 15]
        targ17 = targ_argmax[:, 16]
        targ18 = targ_argmax[:, 17]
        targ19 = targ_argmax[:, 18]
        targ20 = targ_argmax[:, 19]
        targ21 = targ_argmax[:, 20]
        targ22 = targ_argmax[:, 21]
        targ23 = targ_argmax[:, 22]
        targ24 = targ_argmax[:, 23]
        targ25 = targ_argmax[:, 24]
        targ26 = targ_argmax[:, 25]

        pred_1 = predict_argmax[:, 0]
        pred_2 = predict_argmax[:, 1]
        pred_3 = predict_argmax[:, 2]
        pred_4 = predict_argmax[:, 3]
        pred_5 = predict_argmax[:, 4]
        pred_6 = predict_argmax[:, 5]
        pred_7 = predict_argmax[:, 6]
        pred_8 = predict_argmax[:, 7]
        pred_9 = predict_argmax[:, 8]
        pred_10 = predict_argmax[:, 9]
        pred_11 = predict_argmax[:, 10]
        pred_12 = predict_argmax[:, 11]
        pred_13 = predict_argmax[:, 12]
        pred_14 = predict_argmax[:, 13]
        pred_15 = predict_argmax[:, 14]
        pred_16 = predict_argmax[:, 15]
        pred_17 = predict_argmax[:, 16]
        pred_18 = predict_argmax[:, 17]
        pred_19 = predict_argmax[:, 18]
        pred_20 = predict_argmax[:, 19]
        pred_21 = predict_argmax[:, 20]
        pred_22 = predict_argmax[:, 21]
        pred_23 = predict_argmax[:, 22]
        pred_24 = predict_argmax[:, 23]
        pred_25 = predict_argmax[:, 24]
        pred_26 = predict_argmax[:, 25]

        ICC31_au1 = ICC31(targ1, pred_1)
        ICC31_au2 = ICC31(targ2, pred_2)
        ICC31_au3 = ICC31(targ3, pred_3)
        ICC31_au4 = ICC31(targ4, pred_4)
        ICC31_au5 = ICC31(targ5, pred_5)
        ICC31_au6 = ICC31(targ6, pred_6)
        ICC31_au7 = ICC31(targ7, pred_7)
        ICC31_au8 = ICC31(targ8, pred_8)
        ICC31_au9 = ICC31(targ9, pred_9)
        ICC31_au10 = ICC31(targ10, pred_10)
        ICC31_au11 = ICC31(targ11, pred_11)
        ICC31_au12 = ICC31(targ12, pred_12)
        ICC31_au13 = ICC31(targ13, pred_13)
        ICC31_au14 = ICC31(targ14, pred_14)
        ICC31_au15 = ICC31(targ15, pred_15)
        ICC31_au16 = ICC31(targ16, pred_16)
        ICC31_au17 = ICC31(targ17, pred_17)
        ICC31_au18 = ICC31(targ18, pred_18)
        ICC31_au19 = ICC31(targ19, pred_19)
        ICC31_au20 = ICC31(targ20, pred_20)
        ICC31_au21 = ICC31(targ21, pred_21)
        ICC31_au22 = ICC31(targ22, pred_22)
        ICC31_au23 = ICC31(targ23, pred_23)
        ICC31_au24 = ICC31(targ24, pred_24)
        ICC31_au25 = ICC31(targ25, pred_25)
        ICC31_au26 = ICC31(targ26, pred_26)
        ICC31_mean = ICC31(targ_argmax, predict_argmax)

        logs.update({"ICC31_mean": ICC31_mean, "ICC31_au1": ICC31_au1, "ICC31_au2": ICC31_au2, "ICC31_au4": ICC31_au3,
                     "ICC31_au5": ICC31_au4, "ICC31_au6": ICC31_au5, "ICC31_au7": ICC31_au6, "ICC31_au9": ICC31_au7,
                     "ICC31_au10": ICC31_au8, "ICC31_au11": ICC31_au9, "ICC31_au12": ICC31_au10,
                     "ICC31_au14": ICC31_au11,
                     "ICC31_au15": ICC31_au12, "ICC31_au16": ICC31_au13, "ICC31_au17": ICC31_au14,
                     "ICC31_au18": ICC31_au15,
                     "ICC31_au20": ICC31_au16, "ICC31_au22": ICC31_au17, "ICC31_au23": ICC31_au18,
                     "ICC31_au24": ICC31_au19,
                     "ICC31_au25": ICC31_au20, "ICC31_au26": ICC31_au21, "ICC31_au27": ICC31_au22,
                     "ICC31_au28": ICC31_au23, "ICC31_au34": ICC31_au24, "ICC31_au38": ICC31_au25,
                     "ICC31_au43": ICC31_au26})
        print(ICC31_mean, 'ICC31_mean')
        return

def randon_rectangles_occluder_RAF_DB(image):
    """take the image and generate random three rectangle on image
    return: occluded image"""
    x1 = np.random.randint(10,90)
    y1 = np.random.randint(10,90)
    x2 = np.random.randint(10,90)
    y2 = np.random.randint(10,90)
    x3 = np.random.randint(10,90)
    y3 = np.random.randint(10,90)

    a = np.random.randint(10, 30)
    b = np.random.randint(10, 30)
    c = np.random.randint(10, 30)
    d = np.random.randint(10, 30)
    e = np.random.randint(10, 30)
    f = np.random.randint(10, 30)

    occluded_1 = cv2.rectangle(image, (x1,y1), (x1+a,y1+b), (0,0,0),-1 )
    occluded_2 = cv2.rectangle(occluded_1, (x2,y2), (x2+c,y2+d), (0,0,0),-1 )
    occluded_3 = cv2.rectangle(occluded_2, (x3,y3), (x3+e,y3+f), (0,0,0),-1 )
    return occluded_3

def get_random_image_mask(filepath, mask_size):
    """
    filepath: absulate filepath where images are stored in single folder
    mask_size: mask image size, ex (30, 30)
    return image mask to occlude another image
    """
    random_filename = random.choice([x for x in os.listdir(filepath)])
    im = cv2.imread(os.path.join(filepath, random_filename))
    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    roi = img_rgb[50:250, 100:300]
    crop = cv2.resize(roi, mask_size)
    return crop

def fixed_rectangles_occluder(image):
    """take the image and generate random three rectangle on image
    return: occluded image"""
    mask_size = 120
    x1 = np.random.randint(0,128 - mask_size)
    y1 = np.random.randint(0,128 - mask_size)
    a = x1 + mask_size
    b = y1 + mask_size
    occluded_1 = cv2.rectangle(image, (x1, y1), (a, b), (0,0,0),-1 )
    return occluded_1

def eyes_occluder_raf_db(image):
    occluded_1 = cv2.rectangle(image, (10,20), (40,40), (0,0,0),-1 )
    occluded_2 = cv2.rectangle(occluded_1, (60,20), (90,40), (0,0,0),-1 )
    return occluded_2

def fixed_raf_db_face_occluder(image):
    occluded_1 = cv2.rectangle(image, (10,20), (40,40), (0,0,0),-1 )
    occluded_2 = cv2.rectangle(occluded_1, (60,20), (90,40), (0,0,0),-1 )
    occluded_3 = cv2.rectangle(occluded_2, (30,50), (60,65), (0,0,0),-1 )
    occluded_4 = cv2.rectangle(occluded_3, (25,67), (70,85), (0,0,0),-1 )
    return occluded_4

def bosphorus_face_parts_occluder(image):
    occluded_1 = cv2.rectangle(image, (7, 20), (42, 43), (0, 0, 0), -1)
    occluded_2 = cv2.rectangle(occluded_1, (55, 20), (90, 43), (0, 0, 0), -1)
    occluded_3 = cv2.rectangle(occluded_2, (30, 50), (70, 80), (0, 0, 0), -1)
    return occluded_3

def half_face_occluder(image):
    occluded = cv2.rectangle(image, (0,50), (100,100), (0,0,0),-1)
    return occluded

class Metrics_single_AU(keras.callbacks.Callback):
    """
    custom metrics for single action unit
    """
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        predict_argmax = argmax(predict, axis=1)
        targ_argmax = argmax(targ, axis=1)
        self.ICC31=ICC31(targ_argmax, predict_argmax)
        logs.update({"ICC31_au": self.ICC31})
        print(self.ICC31, 'val_ICC31')
        return self.ICC31, logs

def rectangle_occluder(image):
    """
    :param image:
    :return: black bar occluded image
    """
    mask_size = 60
    x1 = np.random.randint(0,100 - mask_size)
    y1 = np.random.randint(0,100 - mask_size)
    x2 = x1 + mask_size
    y2 = y1 + mask_size
    occluded_1 = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0),-1 )
    return occluded_1

def two_rectangles_occluder(image):
    """
    :param image:
    :return: two black bar occluded image
    """
    mask_size = 40
    x1, x2 = random_state_1.randint(0, 100 - mask_size, size=2)
    y1, y2 = random_state_2.randint(0, 100 - mask_size, size=2)
    a = x1 + mask_size
    b = y1 + mask_size
    c = x2 + mask_size
    d = y2 + mask_size
    occluded_1 = cv2.rectangle(image, (x1, y1), (a, b), (0,0,0),-1)
    occluded_2 = cv2.rectangle(occluded_1, (x2, y2), (c, d), (0,0,0),-1)
    return occluded_2

def three_rectangles_occluder(image):
    """
    :param image:
    :return: three black bar occluded image
    """
    mask_size = 25
    x1, x2, x3 = random_state_1.randint(0, 100 - mask_size, size=3)
    y1, y2, y3 = random_state_2.randint(0, 100 - mask_size, size=3)
    a = x1 + mask_size
    b = y1 + mask_size
    c = x2 + mask_size
    d = y2 + mask_size
    e = x3 + mask_size
    f = y3 + mask_size
    occluded_1 = cv2.rectangle(image, (x1, y1), (a, b), (0,0,0),-1)
    occluded_2 = cv2.rectangle(occluded_1, (x2, y2), (c, d), (0,0,0),-1)
    occluded_3 = cv2.rectangle(occluded_2, (x3, y3), (e, f), (0, 0, 0), -1)
    return occluded_3

def random_rectangles_occluder(image):
    """
    :param image:
    :return: two black bar occluded image
    """
    mask_size = 40
    x1, x2 = random_state_x1.randint(0, 100 - mask_size, size=2)
    y1, y2 = random_state_y1.randint(0, 100 - mask_size, size=2)
    a = x1 + mask_size
    b = y1 + mask_size
    c = x2 + mask_size
    d = y2 + mask_size
    occluded_1 = cv2.rectangle(image, (x1, y1), (a, b), (0, 0, 0), -1)
    occluded_2 = cv2.rectangle(occluded_1, (x2, y2), (c, d), (0, 0, 0), -1)
    return occluded_2

def get_segmentation_mask(img):
    """
    :param image:
    :return: seg label mask for two black bar occluded image same as original occlusion
    """
    mask_size = 40
    x1, x2 = random_state_x2.randint(0, 100 - mask_size, size=2)
    y1, y2 = random_state_y2.randint(0, 100 - mask_size, size=2)
    a = x1 + mask_size
    b = y1 + mask_size
    c = x2 + mask_size
    d = y2 + mask_size
    occluded_1 = cv2.rectangle(img, (x1, y1), (a, b), (0, 0, 0), -1)
    gray = cv2.rectangle(occluded_1, (x2, y2), (c, d), (0, 0, 0), -1)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > 0:
            gray_r[i] = 0
        else:
            gray_r[i] = 1
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    gray_ = np.expand_dims(gray, axis=2)
    gray_ = gray_.astype('bool')
    return gray_









