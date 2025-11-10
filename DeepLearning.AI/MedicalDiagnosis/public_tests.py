import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from keras import backend
from test_case import *
from IPython.display import display

### ex1
def check_for_leakage_test(target):
    df1 = pd.DataFrame({'patient_id': [0, 1, 2]})
    df2 = pd.DataFrame({'patient_id': [2, 3, 4]})
    expected_output_1 = True
    
    print("Test Case 1\n")
    print("df1")
    print(df1)
    print("df2")
    print(df2)
    print("leakage output:", target(df1, df2, 'patient_id'), "\n-------------------------------------")
    
    df3 = pd.DataFrame({'patient_id': [0, 1, 2]})
    df4 = pd.DataFrame({'patient_id': [3, 4, 5]})
    expected_output_2 = False
    
    print("Test Case 2\n")
    print("df1") ### same heading for df3
    print(df3)
    print("df2") ### same heading for df4
    print(df4)
    print("leakage output:", target(df3, df4, 'patient_id'), "\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [df1, df2, 'patient_id'],
            "expected": expected_output_1,
            "error":"Data-type mismatch, make sure you are using pandas functions"
        },
        {
            "name":"datatype_check",
            "input": [df3, df4, 'patient_id'],
            "expected": expected_output_2,
            "error":"Datatype mismatch, make sure you are using pandas functions"
        },
        {
            "name": "shape_check",
            "input": [df1, df2, 'patient_id'],
            "expected": expected_output_1,
            "error": "Wrong shape, make sure you are using pandas functions"
        },
        {
            "name": "shape_check",
            "input": [df3, df4, 'patient_id'],
            "expected": expected_output_2,
            "error": "Wrong shape, make sure you are using pandas functions"
        },
        {
            "name": "equation_output_check",
            "input": [df1, df2, 'patient_id'],
            "expected": expected_output_1,
            "error": "Wrong output, make sure you are using pandas functions"
        },
        {
            "name": "equation_output_check",
            "input": [df3, df4, 'patient_id'],
            "expected": expected_output_2,
            "error": "Wrong output, make sure you are using pandas functions"
        }
    ]

    multiple_test(test_cases, target)
    
### ex2
def compute_class_freqs_test(target):
    labels_matrix = np.array(
        [[1, 0, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 1],
         [1, 0, 1]]
    )

    print("Labels:")
    print(labels_matrix)
    pos_freqs, neg_freqs = target(labels_matrix)
    print("\nPos Freqs: ", pos_freqs)
    print("Neg Freqs: ", neg_freqs, "\n")
    
    expected_freqs = (np.array([0.8, 0.4, 0.8]), np.array([0.2, 0.6, 0.2]))
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [labels_matrix],
            "expected": expected_freqs,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [labels_matrix],
            "expected": expected_freqs,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [labels_matrix],
            "expected": expected_freqs,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    
### ex3
def get_weighted_loss_test(target, epsilon):
    y_true, w_p, w_n, y_pred_1, y_pred_2 = get_weighted_loss_test_case()
    
    print("y_true:")
    print(y_true)
    print("\nw_p:")
    print(w_p)
    print("\nw_n:")
    print(w_n)
    print("\ny_pred_1:")
    print(y_pred_1)
    print("\ny_pred_2:")
    print(y_pred_2)
    
    L = target(w_p, w_n, epsilon)
    L1 = L(y_true, y_pred_1).numpy()
    L2 = L(y_true, y_pred_2).numpy()
    
    print("\nIf you weighted them correctly, you'd expect the two losses to be the same.")
    print("With epsilon = 1, your losses should be, L(y_pred_1) = -0.4956203 and L(y_pred_2) = -0.4956203\n")
    print("Your outputs:\n")
    print("L(y_pred_1) = ", L1)
    print("L(y_pred_2) = ", L2)
    print("Difference: L(y_pred_1) - L(y_pred_2) = ", L1-L2, "\n")
    
    expected_output_1 = np.float32(-0.4956203)
    expected_output_2 = np.float32(-0.4956203)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_true, y_pred_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch. Make sure it is a np.float32 value."
        },
        {
            "name":"datatype_check",
            "input": [y_true, y_pred_2],
            "expected": expected_output_2,
            "error": "Data-type mismatch. Make sure it is a np.float32 value."
        },
        {
            "name": "shape_check",
            "input": [y_true, y_pred_1],
            "expected": expected_output_1,
            "error": "Wrong shape."
        },
        {
            "name": "shape_check",
            "input": [y_true, y_pred_2],
            "expected": expected_output_2,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [y_true, y_pred_1],
            "expected": expected_output_1,
            "error": "Wrong output. One possible mistake, your epsilon is not equal to 1."
        },
        {
            "name": "equation_output_check",
            "input": [y_true, y_pred_2],
            "expected": expected_output_2,
            "error": "Wrong output. One possible mistake, your epsilon is not equal to 1."
        }
    ]
    multiple_test_weight_loss(test_cases, L)

### ex1
def get_tp_tn_fp_fn_test(target_1, target_2, target_3, target_4):
    threshold = 0.5
    
    df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                       'preds_test': [0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0],
                       'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                      })
    
    y_test = df['y_test']
    preds_test = df['preds_test']
    
    display(df)
    print(f"""Your functions calcualted: 
    TP: {target_1(y_test, preds_test, threshold)}
    TN: {target_2(y_test, preds_test, threshold)}
    FP: {target_3(y_test, preds_test, threshold)}
    FN: {target_4(y_test, preds_test, threshold)}
    """)
    
    expected_output_1 = np.int64(sum(df['category'] == 'TP'))
    expected_output_2 = np.int64(sum(df['category'] == 'TN'))
    expected_output_3 = np.int64(sum(df['category'] == 'FP'))
    expected_output_4 = np.int64(sum(df['category'] == 'FN'))
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in true_positives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in true_positives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in true_positives"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in true_negatives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in true_negatives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in true_negatives"
        }
    ]
    
    multiple_test(test_cases_2, target_2)
    
    test_cases_3 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Data-type mismatch in false_positives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Wrong shape in false_positives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Wrong output in false_positives"
        }
    ]
    
    multiple_test(test_cases_3, target_3)
    
    test_cases_4 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Data-type mismatch in false_negatives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Wrong shape in false_negatives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Wrong output in false_negatives"
        }
    ]
    
    multiple_test(test_cases_4, target_4)

### ex2
def get_accuracy_test(target):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t  ", y_test)
    print("Test Predictions: ", preds_test)
    print("Threshold:\t  ", threshold)
    print("Computed Accuracy:", target(y_test, preds_test, threshold), "\n")
    
    expected_output = np.float64(0.6)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)

### ex3
def get_prevalence_test(target):
    y_test = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    
    print("Test Case:\n")
    print("Test Labels:\t     ", y_test)
    print("Computed Prevalence: ", target(y_test), "\n")
    
    expected_output = np.float64(0.4)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)

### ex4
def get_sensitivity_specificity_test(target_1, target_2):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t      ", y_test)
    print("Test Predictions:     ", y_test)
    print("Threshold:\t      ", threshold)
    print("Computed Sensitivity: ", target_1(y_test, preds_test, threshold))
    print("Computed Specificity: ", target_2(y_test, preds_test, threshold), "\n")
    
    expected_output_1 = np.float64(0.6666666666666666)
    expected_output_2 = np.float64(0.5)
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in get_sensitivity."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in get_sensitivity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in get_sensitivity"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in get_specificity"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in get_specificity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in get_specificity"
        }
    ]
    
    multiple_test(test_cases_2, target_2)

### ex5
def get_ppv_npv_test(target_1, target_2):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t  ", y_test)
    print("Test Predictions: ", y_test)
    print("Threshold:\t  ", threshold)
    print("Computed PPV:\t  ", target_1(y_test, preds_test, threshold))
    print("Computed NPV:\t  ", target_2(y_test, preds_test, threshold),"\n")
    
    expected_output_1 = np.float64(0.6666666666666666)
    expected_output_2 = np.float64(0.5)
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in get_ppv."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in get_ppv"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in get_ppv"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in get_specificity"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in get_specificity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in get_specificity"
        }
    ]
    
    multiple_test(test_cases_2, target_2)


### ex1
def get_sub_volume_test(target):
    np.random.seed(3)
    image, label = get_sub_volume_test_case()

    print("Image:")
    for k in range(3):
        print(f"z = {k}")
        print(image[:, :, k, 0])

    print("\n")
    print("Label:")
    for k in range(3):
        print(f"z = {k}")
        print(label[:, :, k])
        
    print("\033[1m\nExtracting (2, 2, 2) sub-volume\n\033[0m")
    
    orig_x = 4
    orig_y = 4
    orig_z = 3
    output_x = 2
    output_y = 2
    output_z = 2
    num_classes = 3
    
    expected_output = (np.array([[[[1., 2.],
                                   [2., 4.]],
                                  [[2., 4.],
                                   [4., 8.]]]]), 
                       np.array([[[[1., 0.],
                                   [1., 0.]],
                                  [[1., 0.],
                                   [1., 0.]]],
                                 [[[0., 1.],
                                   [0., 1.]],
                                  [[0., 1.],
                                   [0., 1.]]]], dtype=np.float32))
    
    test_cases = [
         {
             "name":"datatype_check",
             "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
             "expected": expected_output,
             "error": "Data-type mismatch."
         },
        {
            "name": "shape_check",
            "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    learner_func_sample_image, learner_func_sample_label = multiple_test_get_sub_volume(test_cases, target)
    
    print("\033[0m\nSampled Image:")
    for k in range(2):
        print("z = " + str(k))
        print(learner_func_sample_image[0, :, :, k])
        
    print("\nSampled Label:")
    for c in range(2):
        print("class = " + str(c))
        for k in range(2):
            print("z = " + str(k))
            print(learner_func_sample_label[c, :, :, k])
    


##############################################        
### ex2
def standardize_test(target, X):
    X_norm = target(X)
    
    def return_x_norm_value(X_norm): 
        return X_norm[0,:,:,0].std()
    
    print("stddv for X_norm[0, :, :, 0]: ", return_x_norm_value(X_norm), "\n")
    
    expected_output = np.float64(0.9999999999999999)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, return_x_norm_value)
    


##############################################        
### ex3
def single_class_dice_coefficient_test(target, epsilon):
    pred_1, label_1, pred_2, label_2 = single_class_dice_coefficient_test_case()
        
    expected_output_1 = np.float64(0.6)
    expected_output_2 = np.float64(0.8333333333333334) 
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[:, :, 0])
    print("\nLabel:\n")
    print(label_1[:, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nDice coefficient: ", dc_1.numpy(), "\n\n----------------------\n")
        
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[:, :, 0])
    print("\nLabel:\n")
    print(label_2[:, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nDice coefficient: ", dc_2.numpy(), "\n")
        
    axis = (0, 1, 2)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        }
    ]
    
    multiple_test_dice(test_cases, target)
        

##############################################    
### ex4
def dice_coefficient_test(target, epsilon):
    pred_1, label_1, pred_2, label_2, pred_3, label_3 = dice_coefficient_test_case()
        
    expected_output_1 = np.float64(0.6)
    expected_output_2 = np.float64(0.8333333333333334)
    expected_output_3 = np.float64(0.7166666666666667) 
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nDice coefficient: ", dc_1.numpy(), "\n\n----------------------\n")
        
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nDice coefficient: ", dc_2.numpy(), "\n\n----------------------\n")
        
    print("Test Case 3:\n")
    print("Pred:\n")
    print("class = 0")
    print(pred_3[0, :, :, 0], "\n")
    print("class = 1")
    print(pred_3[1, :, :, 0], "\n")
    print("Label:\n")
    print("class = 0")
    print(label_3[0, :, :, 0], "\n")
    print("class = 1")
    print(label_3[1, :, :, 0], "\n")

    dc_3 = target(label_3, pred_3, epsilon=epsilon)
    print("Dice coefficient: ", dc_3.numpy(), "\n")
        
    axis = (1, 2, 3)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Data-type mismatch for Test Case 3"
        },
        {
            "name": "shape_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Wrong shape for Test Case 3"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3. One possible reason for error: make sure epsilon = 1"
        }
    ]

    multiple_test_dice(test_cases, target)
        
##############################################         
### ex5
def soft_dice_loss_test(target, epsilon):
    pred_1, label_1, pred_2, label_2, pred_3, label_3, pred_4, label_4, pred_5, label_5, pred_6, label_6 = soft_dice_loss_test_case()
    
    expected_output_1 = np.float64(0.4)
    expected_output_2 = np.float64(0.4285714285714286)
    expected_output_3 = np.float64(0.16666666666666663)
    expected_output_4 = np.float64(0.006024096385542355)
    expected_output_5 = np.float64(0.21729776247848553)
    expected_output_5 = np.float64(0.21729776247848553)
    expected_output_6 = np.float64(0.4375)
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_1.numpy(), "\n\n----------------------\n")
    
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_2.numpy(), "\n\n----------------------\n")
    
    print("Test Case 3:\n")
    print("Pred:\n")
    print(pred_3[0, :, :, 0])
    print("\nLabel:\n")
    print(label_3[0, :, :, 0])

    dc_3= target(pred_3, label_3, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_3.numpy(), "\n\n----------------------\n")
    
    print("Test Case 4:\n")
    print("Pred:\n")
    print(pred_4[0, :, :, 0])
    print("\nLabel:\n")
    print(label_4[0, :, :, 0])

    dc_4= target(pred_4, label_4, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_4.numpy(), "\n\n----------------------\n")
    
    print("Test Case 5:\n")
    print("Pred:\n")
    print("class = 0")
    print(pred_5[0, :, :, 0], "\n")
    print("class = 1")
    print(pred_5[1, :, :, 0], "\n")
    print("Label:\n")
    print("class = 0")
    print(label_5[0, :, :, 0], "\n")
    print("class = 1")
    print(label_5[1, :, :, 0], "\n")

    dc_5 = target(label_5, pred_5, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_5.numpy(), "\n\n----------------------\n")
    
    print("Test Case 6:\n")
    dc_6 = target(label_6, pred_6, epsilon=epsilon)
    print("Soft Dice Loss: ", dc_6.numpy(), "\n")
        
    axis = (1, 2, 3)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
             "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
             "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
             "error": "Wrong output for Test Case 3. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_4, label_4, axis, epsilon],
            "expected": expected_output_4,
             "error": "Wrong output for Test Case 4. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
            "error": "Data-type mismatch for Test Case 5."
        },
        {
            "name": "shape_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
            "error": "Wrong shape for Test Case 5."
        },
        {
            "name": "equation_output_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
             "error": "Wrong output for Test Case 5. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
            "error": "Data-type mismatch for Test Case 6."
        },
        {
            "name": "shape_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
            "error": "Wrong shape for Test Case 6."
        },
        {
            "name": "equation_output_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
             "error": "Wrong output for Test Case 6. One possible reason for error: make sure epsilon = 1"
        },        
    ]
   
    multiple_test_dice(test_cases, target)
    
##############################################    
### ex6    
def compute_class_sens_spec_test(target):
    pred_1, label_1, pred_2, label_2, df = compute_class_sens_spec_test_case()
    
    expected_output_1 = np.array((0.5, 0.5))
    expected_output_2 = np.array(((0.6666666666666666, 1.0)))
    expected_output_3 = np.array(((0.2857142857142857, 0.42857142857142855)))
    
    
    sensitivity_1, specificity_1 = target(pred_1, label_1, 0)
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])
    print("\nSensitivity: ", sensitivity_1)
    print("Specificity: ", specificity_1, "\n\n----------------------\n")
    
    sensitivity_2, specificity_2 = target(pred_2, label_2, 0)
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])
    print("\nSensitivity: ", sensitivity_2)
    print("Specificity: ", specificity_2, "\n\n----------------------\n")
    
    print("Test Case 3:")
    display(df)
    pred_3 = np.array( [df['preds_test']])
    label_3 = np.array( [df['y_test']])
    sensitivity_3, specificity_3 = target(pred_3, label_3, 0)
    print("\nSensitivity: ", sensitivity_3)
    print("Specificity: ", specificity_3, "\n")
    
    test_cases = [
        {
             "name":"datatype_check",
             "input": [pred_1, label_1, 0],
             "expected": expected_output_1,
             "error": "Data-type mismatch for Test Case 1"
         },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, 0],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, 0],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, 0],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, 0],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3"
        }
    ]
    multiple_test(test_cases, target)