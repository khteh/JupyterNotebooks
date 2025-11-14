from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.preprocessing import image
import pandas as pd
import numpy as np


def get_mean_std_per_batch(IMAGE_DIR, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        path = IMAGE_DIR + img
        sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std    


def load_image_normalize(path, mean, std, H=320, W=320):
    x = image.load_img(path, target_size=(H, W))
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x


def load_image(path, df, preprocess=True, H = 320, W = 320):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        mean, std = get_mean_std_per_batch(df, H=H, W=W)
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


# LOAD MODEL FROM C1M2
def load_C3M3_model(path):
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

    train_df = pd.read_csv(path + "data/nih_new/train-small.csv")
    valid_df = pd.read_csv(path + "data/nih_new/valid-small.csv")
    test_df = pd.read_csv(path + "data/nih_new/test.csv")

    class_pos = train_df.loc[:, labels].sum(axis=0)
    class_neg = len(train_df) - class_pos
    class_total = class_pos + class_neg

    pos_weights = class_pos / class_total
    neg_weights = class_neg / class_total
    print("Got loss weights")
    # create the base pre-trained model
    base_model = DenseNet121(weights=path+'pretained_model/densenet.hdf5', include_top=False)
    print("Loaded DenseNet")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    print("Added layers")

    model = Model(inputs=base_model.input, outputs=predictions)

    def get_weighted_loss(neg_weights, pos_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
            # L(X, y) = −w * y log p(Y = 1|X) − w *  (1 − y) log p(Y = 0|X)
            # from https://arxiv.org/pdf/1711.05225.pdf
            loss = 0
            for i in range(len(neg_weights)):
                loss -= (neg_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + 
                         pos_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            
            loss = K.sum(loss)
            return loss
        return weighted_loss
    
    model.compile(optimizer='adam', loss=get_weighted_loss(neg_weights, pos_weights))
    print("Compiled Model")

    model.load_weights(path + "data/nih_new/pretrained_model.h5")
    print("Loaded Weights")
    return model