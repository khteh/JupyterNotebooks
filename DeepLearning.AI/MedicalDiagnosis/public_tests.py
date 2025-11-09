import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from keras import backend
from test_case import *

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
def get_weighted_loss_test(target, epsilon, sess):
    y_true, w_p, w_n, y_pred_1, y_pred_2 = get_weighted_loss_test_case(sess)
    
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
    L1 = L(y_true, y_pred_1).eval(session=sess)
    L2 = L(y_true, y_pred_2).eval(session=sess)
    
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
    
    multiple_test_weight_loss(test_cases, L, sess)

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


