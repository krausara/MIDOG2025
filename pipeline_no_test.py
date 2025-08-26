import os
import cv2
import torchstain
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from aucmedi import *
from aucmedi.evaluation import *
from aucmedi.ensemble import Bagging
from aucmedi.neural_network.model import NeuralNetwork
from aucmedi.utils.class_weights import compute_class_weights
from aucmedi.neural_network.loss_functions import categorical_focal_loss
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES']='1' 

# stain normalization parameters
# allowed stainers: 'reinhard', 'macenko'
# allowed stain methods: 'modified' for modified Reinhard else None
stainer = 'midog+octo-test'
stain_method = None
# path to target image for stain normalization
target_path = './data/2935.png' 

# training parameters
architecture = '2D.ConvNeXtBase' 
k = 3
batch_size = 73
epochs = 1000
iterations = 180 

seed = 1234
callbacks = []

# Paths
# path to input images
path_input = 'data'
path_df_train = 'midog+octo.csv'

# path to results directories
path_output = os.path.join(stainer, 'out')
if not os.path.exists(path_output) : os.makedirs(path_output)
path_evaluate = os.path.join(stainer, 'eval')
if not os.path.exists(path_evaluate) : os.makedirs(path_evaluate)

# path for history.csv to save training history
path_history = os.path.join(stainer, 'out', 'history.csv')

# create directory and file to perist models
path_snapshot = os.path.join(stainer, 'out', 'snapshots')
if not os.path.exists(path_output) : os.makedirs(path_snapshot)
path_snapshot_file = os.path.join(stainer, 'out', 'snapshots', 'model.keras')

# stain normalisation subfunction
class stain_norm(Subfunction_Base):
    def __init__(self, method,  target_path):
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='numpy', method=method)
        target_img = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
        self.normalizer.fit(target_img)

    def transform(self, image):
        new_image = self.normalizer.normalize(I=image)
        return new_image

# Pillar 1   
def prepare_data():
    # load training dataset
    ds_loader = input_interface(interface='csv', 
                                path_data=path_df_train, 
                                path_imagedir=path_input,
                                training=True, 
                                ohe=False, 
                                col_sample='image_id', 
                                col_class='majority')
    (train_samples, train_class_ohe, nclasses, class_names, image_format) = ds_loader

    # evaluate training dataset
    df = evaluate_dataset(train_samples, 
                        train_class_ohe, 
                        out_path=path_output, 
                        class_names=class_names, 
                        plot_barplot=True,
                        suffix='train')
    path_txt = os.path.join(path_output, 'class_distr.txt')
    df.to_csv(path_txt, sep = '\t', index=False, mode='a')

    train = (train_samples, train_class_ohe)
    metadata = (nclasses, class_names, image_format)
    return train, metadata

# Pillar 2
def run_aucmedi(train_x, train_y, ds_meta, target_path):
    '''
    Run the AUCmedi pipeline for training a neural network model on the provided dataset.
    This function prepares the data, defines the model, sets up callbacks and trains the model using k-fold cross-validation. 
    It also applies stain normalization to the images using the Reinhard normalizer and provide target image.
    Args:
        train_x (list): List of training sample file paths.
        train_y (list): List of training sample labels.
        ds_meta (tuple): Metadata of the dataset containing number of classes, class names, and image format.
        target_path (str): Path to target image for stain normalization.
    '''
    # unpack dataset meta information 
    (nclasses, _, image_format) = ds_meta

    # compute class weight for loss function
    cw_loss, _ = compute_class_weights(ohe_array=train_y)
    my_loss = categorical_focal_loss(alpha=cw_loss)

    # define image augmentation
    img_aug = ImageAugmentation(flip = True,
                                rotate = True,
                                brightness = True,
                                contrast = True,
                                saturation = True,
                                hue = True,
                                scale = True,
                                crop = False,
                                grid_distortion = False,
                                compression = False,
                                gaussian_noise = True, 
                                gaussian_blur = False, 
                                downscaling = False,
                                gamma = False, 
                                elastic_transform = True)

    # Define Callbacks
    # early stopping, if none or very small improvements in performance, we abort training
    cb_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=32, 
        verbose=1,
    )
    callbacks.append(cb_early_stopping)

    # dynamic learing rate, if improvements start to stagnate reducing lr is beneficial to model 
    cb_dynamic_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        facotr = 0.1,
        patience = 5,
        verbose = 1,
        mode='min',
        min_lr = 1e-7
    )
    callbacks.append(cb_dynamic_lr)

    # persists history of training
    cb_csv_logger = CSVLogger(
        filename=path_history,
        separator=',',
        append=True
    )
    callbacks.append(cb_csv_logger)

    # perists models
    cb_model_checkpoint = ModelCheckpoint(
        filepath=path_snapshot_file,
        monitor='val_loss',
        save_best_only=True
    )
    callbacks.append(cb_model_checkpoint)

    # define model
    model = NeuralNetwork(n_labels=nclasses,
                          channels=3,
                          architecture=architecture,
                          loss=my_loss,
                          pretrained_weights=True)

    # k-fold cross-validation via Bagging object
    el = Bagging(model, k_fold = k) 

    # Pillar 3
    # define trainign data generator
    train_generator = DataGenerator(samples=train_x, 
                                    path_imagedir=path_input,
                                    labels=train_y, 
                                    data_aug=img_aug,
                                    subfunctions=[stain_norm(method=stain_method,
                                                             target_path=target_path)], 
                                    resize=model.meta_input, 
                                    standardize_mode=model.meta_standardize,
                                    image_format=image_format, 
                                    batch_size=batch_size,  
                                    grayscale=False, 
                                    prepare_images=True, 
                                    sample_weights=None, 
                                    seed=seed, 
                                    workers=1) 

    # train model
    el.train(train_generator, epochs=epochs, iterations=iterations, callbacks=callbacks, transfer_learning=True)
 
    # dump latst model
    el.dump(path_output)
    


if __name__ == '__main__':
    # Run the pipeline
    train, ds_meta = prepare_data()
    run_aucmedi(train[0], train[1], ds_meta, target_path)
