import os
import cv2
from libs.config import LABELS_DEER, INV_LABELMAP_DEER, test_ids
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from libs.util_keras import get_segmentation_array, get_iou


def wherecolor(img, color, negate = False):

    k1 = (img[:, :, 0] == color[0])
    k2 = (img[:, :, 1] == color[1])
    k3 = (img[:, :, 2] == color[2])

    if negate:
        return np.where( not (k1 & k2 & k3) )
    else:
        return np.where( k1 & k2 & k3 )

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          savedir="predictions"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    labels_used = unique_labels(y_true, y_pred)
    classes = classes[labels_used]

    # Normalization with generate NaN where there are no ground label labels but there are predictions x/0
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    base, fname = os.path.split(title)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=fname,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlim([-0.5, cm.shape[1] - 0.5])
    plt.ylim([-0.5, cm.shape[0]- 0.5])

    fig.tight_layout()
    # save to directory
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    savefile = title
    # plt.savefig(savefile)
    return savefile, cm,labels_used

def score_masks(labelfile, predictionfile):

    label = cv2.imread(labelfile)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    prediction = cv2.imread(predictionfile)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/1.png', prediction)

    shape = label.shape[:2]

    label_class = np.zeros(shape, dtype='uint8')
    pred_class  = np.zeros(shape, dtype='uint8')

    for color, category in INV_LABELMAP_DEER.items():
        locs = wherecolor(label, color)
        label_class[locs] = category

    for color, category in INV_LABELMAP_DEER.items():
        locs = wherecolor(prediction, color)
        pred_class[locs] = category

    label_class = label_class.reshape((label_class.shape[0] * label_class.shape[1]))
    pred_class = pred_class.reshape((pred_class.shape[0] * pred_class.shape[1]))

    # Remove all predictions where there is a IGNORE (magenta pixel) in the groud label and then shift labels down 1 index
    # not_ignore_locs = np.where(label_class != 0)
    # label_class = label_class[not_ignore_locs] 
    # pred_class = pred_class[not_ignore_locs] -

    precision = precision_score(label_class, pred_class, average='weighted')
    recall = recall_score(label_class, pred_class, average='weighted')
    f1 = f1_score(label_class, pred_class, average='weighted')
    print(f'precision={precision} recall={recall} f1={f1}')

    savefile, cm, labels_used = plot_confusion_matrix(label_class, pred_class, np.array(LABELS_DEER), title=predictionfile.replace(".png","") + "-cf-matrix.png")

    print("CM")
    print(cm)

    return precision, recall, f1, savefile,cm, labels_used

def evaluate(model=None, dataset_path=None,num_classes=None):
    output_height = 320
    output_width = 320

    inp_images = [ dataset_path + '/image-chips/' + fname for fname in os.listdir(os.path.join(dataset_path,'image-chips'))]
    annotations = [ dataset_path + '/label-chips/' + fname for fname in os.listdir(os.path.join(dataset_path,'label-chips'))]

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    n_pixels = np.zeros(num_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        x = cv2.imread(inp)
        pr = model.predict(np.array([x]))[0]
        pr = pr.reshape((output_height,  output_width, num_classes)).argmax(axis=2)
        gt = get_segmentation_array(ann, num_classes,
                                    output_width, output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(num_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }


def score_predictions(dataset, basedir='predictions'):

    num_classes = 4
    count_cf = [0, 0, 0, 0]
    scores = []

    precision = []
    recall = []
    f1 = []

    cf_matrix = np.zeros([num_classes,num_classes])
    predictions = []
    confusions = []

    test_file = open(os.path.join(dataset,'test.txt'),'r')


    #for scene in train_ids + val_ids + test_ids:
    # for scene in test_ids:
    for scene in test_file.readlines():
        if scene.find('.png') < 0:
            continue
        scene = scene.replace("\n","")
        scene = scene.split('.')[0]
        labelfile = f'{dataset}/SegmentationClassPNG/{scene}.png'
        predsfile = os.path.join(basedir, f'{scene}-prediction.png')

        if not os.path.exists(labelfile):
            continue

        if not os.path.exists(predsfile):
            continue

        a, b, c, savefile, cm, labels = score_masks(labelfile, predsfile)
        for i in range(0,len(labels)):
            if not np.isnan(np.sum(cm[i])):
                count_cf[labels[i]] = count_cf[labels[i]] + 1
            for j in range(0,len(labels)):
                ind_i = int(labels[i])
                ind_j = int(labels[j])
                if not np.isnan(cm[i][j]):
                    cf_matrix[ind_i][ind_j] =  cf_matrix[ind_i][ind_j] + cm[i][j]
        precision.append(a)
        recall.append(b)
        f1.append(c)

        predictions.append(predsfile)
        confusions.append(savefile)
    print(count_cf)
    for i in range(0,len(count_cf)):
        if count_cf[i] != 0:
            cf_matrix[i] = cf_matrix[i]/count_cf[i]
    classes = ['tree','ground','water','other']
    title = 'Confusion-Matrix'
    fig, ax = plt.subplots()
    im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cf_matrix.shape[1]),
           yticks=np.arange(cf_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    normalize = True

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cf_matrix.max() / 2.
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            ax.text(j, i, format(cf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cf_matrix[i, j] > thresh else "black")

    plt.xlim([-0.5, cf_matrix.shape[1] - 0.5])
    plt.ylim([-0.5, cf_matrix.shape[0]- 0.5])

    fig.tight_layout()
    # save to directory
    plt.savefig(os.path.join(basedir,title + '.png'))

    # Compute test set scores
    scores = {
        'f1_mean' : np.mean(f1),
        'f1_std'  : np.std(f1),
        'pr_mean' : np.mean(precision),
        'pr_std'  : np.std(precision),
        're_mean' : np.mean(recall),
        're_std'  : np.std(recall),
    }

    return scores, zip(predictions, confusions)


if __name__ == '__main__':
    score_predictions('dataset-sample')
