import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from test_case import *
from IPython.display import display
np.random.seed(3)

### ex1
def make_standard_normal_test(target):
    tmp_train, tmp_test = make_standard_normal_test_case()
    
    print("Tmp Train:\n\n", tmp_train, "\n")
    print("Tmp Test:\n\n", tmp_test, "\n")
    
    tmp_train_transformed, tmp_test_transformed = target(tmp_train,tmp_test)
    
    print("Tmp Train After Standard Normal:\n\n", tmp_train_transformed, "\n")
    print("Tmp Test After Standard Normal:\n\n", tmp_test_transformed, "\n")
    
    print(f"Training set transformed field1 has mean {tmp_train_transformed['field1'].mean(axis=0)} and standard deviation {tmp_train_transformed['field1'].std(axis=0):.4f} ")
    print(f"Test set transformed, field1 has mean {tmp_test_transformed['field1'].mean(axis=0)} and standard deviation {tmp_test_transformed['field1'].std(axis=0):.4f}")
    print(f"Skew of training set field1 before transformation: {tmp_train['field1'].skew(axis=0)}")
    print(f"Skew of training set field1 after transformation: {tmp_train_transformed['field1'].skew(axis=0)}")
    print(f"Skew of test set field1 before transformation: {tmp_test['field1'].skew(axis=0)}")
    print(f"Skew of test set field1 after transformation: {tmp_test_transformed['field1'].skew(axis=0)}\n")
    
    def test_target(tmp_train_transformed, tmp_test_transformed):
        
        return tmp_train_transformed['field1'].mean(axis=0), tmp_test_transformed['field1'].mean(axis=0), tmp_train['field1'].skew(axis=0), tmp_train_transformed['field1'].skew(axis=0), tmp_test['field1'].skew(axis=0), tmp_test_transformed['field1'].skew(axis=0)
        
    expected_output = (np.array([-7.401486830834377e-17, 0.11441332564321975, 1.6523167403329906, 1.0857243344604632, 1.3896361387064917, 0.13709698849045696]))
    
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_train_transformed, tmp_test_transformed],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [tmp_train_transformed, tmp_test_transformed],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_train_transformed, tmp_test_transformed],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, test_target)
    


##############################################        
### ex2
def lr_model_test(target, X_train, y_train):
    tmp_model = target(X_train[0:3], y_train[0:3])
    
    
    def test_target(tmp_model):
        return tmp_model.predict(X_train[4:5]), tmp_model.predict(X_train[5:6])
      
    # Output for learner
    x, y = test_target(tmp_model)
    print('X_train[4:5] value:',x,'\nX_train[5:6] value:',y, "\n")

    expected_output = (np.array([1.]), np.array([1.]))

    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_model],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [tmp_model],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_model],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, test_target)


##############################################        
### ex3
def cindex_test(target):
    y_true, scores_1, scores_2, scores_3 = cindex_test_case()
    
    print("Test Case 1:\n")
    print("Y_true: ", y_true)
    print("Scores: ", scores_1)
    print("cindex for test case 1: ", target(y_true, scores_1))
    
    print("\nTest Case 2:\n")
    print("Y_true: ", y_true)
    print("Scores: ", scores_2)
    print("cindex for test case 2: ", target(y_true, scores_2))
    
    print("\nTest Case 3:\n")
    print("Y_true: ", y_true)
    print("Scores: ", scores_3)
    print("cindex for test case 3: ", target(y_true, scores_3), "\n")
     
    
    expected_output_1 = 0.0
    expected_output_2 = 1.0
    expected_output_3 = 0.875
    
    
    ### test cases
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_true, scores_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for test case 1."
        },
        {
            "name": "shape_check",
            "input": [y_true, scores_1],
            "expected": expected_output_1,
            "error": "Wrong shape for test case 1."
        },
        {
            "name": "equation_output_check",
            "input": [y_true, scores_1],
            "expected": expected_output_1,
            "error": "Wrong output for test case 1."
        },
        {
            "name": "equation_output_check",
            "input": [y_true, scores_2],
            "expected": expected_output_2,
            "error": "Wrong output for test case 2."
        },
        {
            "name": "equation_output_check",
            "input": [y_true, scores_3],
            "expected": expected_output_3,
            "error": "Wrong output for test case 3."
        }
    ]
    
    multiple_test(test_cases, target)
        

##############################################    
### ex4
def add_interactions_test(target, X_train):
    print("Original Data\n")
    print(X_train.loc[:, ['Age', 'Systolic_BP']].head())
    print("\nData with Interactions\n")
    print(target(X_train.loc[:, ['Age', 'Systolic_BP']].head()), "\n")
    
    
    def test_target(X_train,target):
        tmp_df = target(X_train.loc[:, ['Age', 'Systolic_BP']].head())
        return tuple(tmp_df['Age_x_Systolic_BP'])
    
    expected_output = (0.062063703881263754, -0.5193674805355649, 0.4017996725326609, -2.366724661667246, 0.024344196818138278)
        
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X_train,target],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [X_train,target],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [X_train,target],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, test_target)

def fraction_rows_missing_test(target, X_train, X_val, X_test):
    df_test = fraction_rows_missing_test_case()
    
    print("Example dataframe:\n\n", df_test, "\n")
    print("Computed fraction missing: ", target(df_test))
    print("Fraction of rows missing from X_train: ", target(X_train))
    print("Fraction of rows missing from X_val: ", target(X_val))
    print("Fraction of rows missing from X_test: ", target(X_test))
    
    expected_output = 0.75
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)

def frac_censored_test(target, data):
    data = data
    print("Observations which were censored: ", target(data))
    expected_output = 0.325
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [data],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [data],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [data],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex2
def naive_estimator_test(target):
    sample_df_1, sample_df_2 = naive_estimator_test_case()
    
    print("Sample 1 dataframe for testing code:\n")
    print(sample_df_1)
    print("\n")
    
    print("Test Case 1: S(3)")
    print("Output: ", target(3, sample_df_1))

    print("\nTest Case 2: S(12)")
    print("Output: ", target(12, sample_df_1))

    print("\nTest Case 3: S(20)")
    print("Output: ", target(20, sample_df_1))
    
    print("\nSample 2 dataframe for testing code:\n")
    print("\n", sample_df_2, "\n")

    print("Test case 4: S(5)")
    print("Output: ", target(5, sample_df_2), "\n")
    
    expected_output_1 = 1.0
    expected_output_2 = 0.5
    expected_output_3 = 0.0
    expected_output_4 = 0.5
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [12, sample_df_1],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        },
        {
            "name": "equation_output_check",
            "input": [20, sample_df_1],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3."
        },
        {
            "name": "equation_output_check",
            "input": [5, sample_df_2],
            "expected": expected_output_4,
            "error": "Wrong output for Test Case 4."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex3
def HomemadeKM_test(target):
    
    sample_df_1, sample_df_2 = HomemadeKM_test_case()
    
    print("Test Case 1\n")
    print(sample_df_1.head(), "\n")
    x, y = target(sample_df_1)
    print("Test Case 1 Event times: {}, Survival Probabilities: {}".format(x, y))
    
    print("\nTest Case 2\n")
    print(sample_df_2.head(), "\n")
    x, y = target(sample_df_2)
    print("Test Case 2 Event times: {}, Survival Probabilities: {}".format(x, y), "\n")
    
    expected_output_1 = (np.array([0, 5, 10, 15]), np.array([1.0, 1.0, 0.5, 0.5]))
    expected_output_2 = (np.array([0, 2, 10, 12, 15, 20]), np.array([1.0, 1.0, 0.75, 0.5, 0.5, 0.0]))
    
    test_cases = [
        
        {
            "name": "shape_check",
            "input": [sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [sample_df_2],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        }
    ]
    
    multiple_test(test_cases, target)
    
    
    