from keras.metrics import Metric
from keras import backend as K
from tensorflow.python.keras.utils import metrics_utils
import numpy as np
import os
import cv2 
import six

EPS = 1e-12
# adapted from keras.metrics.Precision
class FBeta(Metric):
    def __init__(self,
                 beta=1,
                 name=None,
                 dtype=None):
        super(FBeta, self).__init__(name=name, dtype=dtype)
        self.beta2 = beta*beta
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(1,),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(1,),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(1,),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[metrics_utils.NEG_INF],
            top_k=1,
            class_id=None,
            sample_weight=sample_weight)

    def _precision(self):
        denom = (self.true_positives + self.false_positives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))
        return result[0]

    def _recall(self):
        denom = (self.true_positives + self.false_negatives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))
        return result[0]

    def result(self):
        precision, recall = self._precision(), self._recall()
        denom = self.beta2 * precision + recall
        result = K.switch(
            K.greater(denom, 0),
            (1 + self.beta2) * precision * recall / denom,
            0.)
        return result

    def reset_states(self):
        K.batch_set_value(
            [(v, np.zeros((1,))) for v in self.weights])

    def get_config(self):
        config = {
            'beta2': self.beta2
        }
        base_config = super(FBeta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DataLoaderError(Exception):
    pass


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels

def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val