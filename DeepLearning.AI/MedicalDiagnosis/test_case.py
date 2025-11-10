import numpy as np, tensorflow as tf, pandas as pd
from keras import backend as K

### ex3
def get_weighted_loss_test_case():
    y_true = tf.constant(np.array(
        [[1, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 1]]
    ))
    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    
    y_pred_1 = tf.constant(0.7*np.ones(y_true.shape))
    y_pred_2 = tf.constant(0.3*np.ones(y_true.shape))
    return y_true.numpy(), w_p, w_n, y_pred_1.numpy(), y_pred_2.numpy()

### ex1
def get_sub_volume_test_case():
    np.random.seed(3)
    
    image = np.zeros((4, 4, 3, 1))
    label = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                image[i, j, k, 0] = i*j*k
                label[i, j, k] = k
                
    return image, label

### ex3
def single_class_dice_coefficient_test_case():
    pred_1 = np.expand_dims(np.eye(2), -1)
    label_1 = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)
    
    pred_2 = np.expand_dims(np.eye(2), -1)
    label_2 = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)
        
    return pred_1, label_1, pred_2, label_2

### ex4
def dice_coefficient_test_case():
    pred_1 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_1 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)
    
    pred_2 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_2 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)
    
    pred_3 = np.zeros((2, 2, 2, 1))
    pred_3[0, :, :, :] = np.expand_dims(np.eye(2), -1)
    pred_3[1, :, :, :] = np.expand_dims(np.eye(2), -1)

    label_3 = np.zeros((2, 2, 2, 1))
    label_3[0, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)
    label_3[1, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)
        
    return pred_1, label_1, pred_2, label_2, pred_3, label_3

### ex5
def soft_dice_loss_test_case():
    pred_1 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_1 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)
    
    pred_2 = np.expand_dims(np.expand_dims(0.5*np.eye(2), 0), -1)
    label_2 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)
    
    pred_3 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_3 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)
    
    pred_4 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    pred_4[0, 0, 1, 0] = 0.8
    label_4 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)
    
    pred_5 = np.zeros((2, 2, 2, 1))
    pred_5[0, :, :, :] = np.expand_dims(0.5*np.eye(2), -1)
    pred_5[1, :, :, :] = np.expand_dims(np.eye(2), -1)
    pred_5[1, 0, 1, 0] = 0.8
    label_5 = np.zeros((2, 2, 2, 1))
    label_5[0, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)
    label_5[1, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)
    
    pred_6 = np.array([
        [
            [
                [1.0, 1.0], [0.0, 0.0]
            ],
            [
                [1.0, 0.0], [0.0, 1.0]
            ]
        ],
        [
            [
                [1.0, 1.0], [0.0, 0.0]
            ],
            [
                [1.0, 0.0], [0.0, 1.0]
            ]
        ],
    ])

    label_6 = np.array([
        [
            [
                [1.0, 0.0], [1.0, 0.0]
            ],
            [
                [1.0, 0.0], [0.0, 0.0]
            ]
        ],
        [
            [
                [0.0, 0.0], [0.0, 0.0]
            ],
            [
                [1.0, 0.0], [0.0, 0.0]
            ]
        ]
    ])
        
    return pred_1, label_1, pred_2, label_2, pred_3, label_3, pred_4, label_4, pred_5, label_5, pred_6, label_6

### ex6
def compute_class_sens_spec_test_case():
    pred_1 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_1 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)
    
    pred_2 = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label_2 = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)
    
    df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                       'preds_test': [1,1,0,0,0,1,1,1,1,0,0,0,0,0],
                       'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                      })
    
    return pred_1, label_1, pred_2, label_2, df
