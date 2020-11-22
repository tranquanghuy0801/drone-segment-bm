from keras import optimizers, metrics
from libs import datasets_keras
from libs.util_keras import FBeta
from libs.losses import categorical_focal_loss
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import wandb
from wandb.keras import WandbCallback

def train_model(dataset, model):
    epochs = 15
#     epochs = 0
    lr     = 1e-4
    size   = 300
    wd     = 1e-2
    bs     = 8 # reduce this if you are running out of GPU memory
    pretrained = True
    alpha_fl = [0.4,0.4,0.15,0.05]
    gamma_fl = 2

    config = {
        'epochs' : epochs,
        'lr' : lr,
        'size' : size,
        'wd' : wd,
        'bs' : bs,
        'alpha_fl': alpha_fl,
        'gamma_fl': gamma_fl,
        'pretrained' : pretrained
    }

    wandb.config.update(config)

    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=[categorical_focal_loss(alpha=alpha_fl,gamma=gamma_fl)],
        metrics=[
            metrics.Precision(top_k=1, name='precision'),
            metrics.Recall(top_k=1, name='recall'),
            metrics.Accuracy(name='accuracy')
        ]
    )
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 7, 
        mode        = 'min', 
        verbose     = 1
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )

    train_data, valid_data = datasets_keras.load_dataset(dataset, bs)
    _, ex_data = datasets_keras.load_dataset(dataset, 10)
    model.fit_generator(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=[
            early_stop,
            reduce_on_plateau,
            WandbCallback(
                input_type='image',
                output_type='segmentation_mask',
                validation_data=ex_data[0]
            )
        ]
    )
