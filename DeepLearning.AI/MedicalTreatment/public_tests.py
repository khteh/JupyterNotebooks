import numpy as np
import pandas as pd
import tensorflow as tf
from test_utils import *
from test_case import *
np.random.seed(3)

### ex1
def proportion_treated_test(target):
    example_df_1, example_df_2, example_df_3 = proportion_treated_test_case()
    
    print("Test Case 1:\n\nExample df:\n")
    print(example_df_1)
    expected_output_1 = np.float64(0.75)
    print(f"Proportion of patient treated: {target(example_df_1)}\n")
    
    print("Test Case 2:\n\nExample df:\n")
    print(example_df_2)
    expected_output_2 = np.float64(0.0)
    print(f"Proportion of patient treated: {target(example_df_2)}\n")
    
    print("Test Case 3:\n\nExample df:\n")
    print(example_df_3)
    expected_output_3 = np.float64(0.25)
    print(f"Proportion of patient treated: {target(example_df_3)}\n")

    test_cases = [
        {
            "name":"datatype_check",
            "input": [example_df_1],
            "expected": expected_output_1,
            "error":"Data-type mismatch."
        },  
        {
            "name": "shape_check",
            "input": [example_df_1],
            "expected": expected_output_1,
            "error": "Wrong shape for test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [example_df_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [example_df_2],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        },
        {
            "name": "equation_output_check",
            "input": [example_df_3],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3."
        }
    ]
    
    multiple_test(test_cases, target)   
    

##############################################

### ex2
def event_rate_test(target):
    

    example_df_1 = event_rate_test_case()
    
    
    expected_output = np.array([0.5, 0.75])
        
    print("Test Case 1:\n")
    print("Example df: \n", example_df_1)
    
    treated_prob, control_prob = target(example_df_1)
    
    print(f"\nTreated 5-year death rate: {treated_prob}")
    print(f"Control 5-year death rate: {control_prob}\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [example_df_1],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [example_df_1],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [example_df_1],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]

    multiple_test(test_cases, target)
    
    
##############################################

## ex3
def extract_treatment_effect_test(target, lr, X_dev):
    
    theta_TRTMT, trtmt_OR = target(lr, X_dev)
    
    print(f"Theta_TRTMT: {theta_TRTMT}")
    print(f"Treatment Odds Ratio: {trtmt_OR}\n")
    
    expected_output = np.array([-0.2885162279475891, 0.7493746442363785])
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [lr, X_dev],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [lr, X_dev],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [lr, X_dev],
            "expected": expected_output,
            "error": "Wrong output."
        },
    ]
    
    multiple_test(test_cases, target)


##############################################

## ex4
def OR_to_ARR_test(target):
    
    test_p_1, test_OR_1, test_p_2, test_OR_2 = OR_to_ARR_test_case()
    expected_output_1, expected_output_2 = 0.15000000000000002, -0.007619047619047616
    
    print("Test Case 1:\n")
    print(f"baseline p: {test_p_1}, OR: {test_OR_1}")
    print(f"Output: {target(test_p_1, test_OR_1)}\n\n")

    print("Test Case 2:\n")
    print(f"baseline p: {test_p_2}, OR: {test_OR_2}")
    print(f"Output: {target(test_p_2, test_OR_2)}\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [test_p_1, test_OR_1],
            "expected": expected_output_1,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [test_p_1, test_OR_1],
            "expected": expected_output_1,
            "error": "Wrong shape, for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [test_p_1, test_OR_1],
            "expected": expected_output_1,
            "error": "Wrong output, for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [test_p_2, test_OR_2],
            "expected": expected_output_2,
            "error": "Wrong output, for Test Case 2."
        }
    ]
    multiple_test(test_cases, target)
    
    
##############################################    

### ex5
def base_risks_test(target, X_dev, lr):
    
    example_df = pd.DataFrame(columns = X_dev.columns)
    example_df.loc[0, :] = X_dev.loc[X_dev.TRTMT == 1, :].iloc[0, :]
    example_df.loc[1, :] = example_df.iloc[0, :]
    example_df.loc[1, 'TRTMT'] = 0
    
    expected_output = np.array([0.43115868, 0.43115868])

    print("Test Case 1:\n")
    print(example_df)
    print(example_df.loc[:, ['TRTMT']])
    print('\n')

    print("Base risks for both rows should be the same.")
    print(f"Baseline Risks: {target(example_df.copy(deep=True), lr)}\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [example_df.copy(deep=True), lr],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [example_df.copy(deep=True), lr],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [example_df.copy(deep=True), lr],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    multiple_test(test_cases, target)
    

##############################################    

### ex6
def lr_ARR_quantile_test(target, X_dev, y_dev, lr):
    
    data = np.array([0.089744, 0.042857, -0.014604, 0.122222, 0.142857, -0.104072, 0.150000, 0.293706, 0.083333, 0.200000])
    idx = [0.231595, 0.314713, 0.386342 , 0.458883, 0.530568, 0.626937, 0.693404, 0.777353, 0.836617, 0.918884]
    d = {'ARR': data, 'baseline_risk':idx}
    df = pd.DataFrame(d).set_index('baseline_risk')    
    expected_output = df.squeeze()
    
    # Test
    abs_risks = target(X_dev, y_dev, lr)

    # print the Series
    print(abs_risks,'\n')
        
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X_dev, y_dev, lr],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [X_dev, y_dev, lr],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [X_dev, y_dev, lr],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    multiple_test(test_cases, target)
    
    
##############################################    

### ex7
def c_for_benefit_score_test(target):
    print("Test Case 1:\n")
    tmp_pairs = [((0.64, 1), (0.54, 0)), 
                 ((0.44, 0),(0.40, 1)), 
                 ((0.56, 1), (0.74, 0)), 
                 ((0.22,0),(0.22,1)), 
                 ((0.22,1),(0.22,0))]
    print(f"pairs: {tmp_pairs}")
    tmp_cstat = target(tmp_pairs)
    expected_output = 0.75
    print(f"\nOutput: {tmp_cstat}\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_pairs],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [tmp_pairs],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_pairs],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    multiple_test(test_cases, target)


##############################################    

### ex8
def c_statistic_test(target):
    
    print("Test Case:\n")
    tmp_pred_rr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    tmp_y = [0,1,0,1,0,1,0,1,0]
    tmp_w = [0,0,0,0,1,1,1,1,1]
    
    tmp_cstat = target(tmp_pred_rr, tmp_y, tmp_w)
    expected_output = 0.6
    
    print(f"C-for-benefit calculated: {tmp_cstat}\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_pred_rr, tmp_y, tmp_w],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [tmp_pred_rr, tmp_y, tmp_w],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_pred_rr, tmp_y, tmp_w],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    multiple_test(test_cases, target)
    
    
##############################################

### ex10
def holdout_grid_search_test(target, X_dev, y_dev):
    
    expected_output = True
    # Test
    n = X_dev.shape[0]
    tmp_X_train = X_dev.iloc[:int(n*0.8),:]
    tmp_X_val = X_dev.iloc[int(n*0.8):,:]
    tmp_y_train = y_dev[:int(n*0.8)]
    tmp_y_val = y_dev[int(n*0.8):]

    # Note that we set random_state to zero
    # in order to make the output consistent each time it's run.
    hyperparams = {
        'n_estimators': [10, 20],
        'max_depth': [2, 5],
        'min_samples_leaf': [0.1, 0.2],
        'random_state': [0]
    }
        
    from sklearn.ensemble import RandomForestClassifier
    control_model = target(RandomForestClassifier, tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val, hyperparams, verbose=True)

    value = (str(control_model)).find("best_score: 0.5928")
    
    def target_value_test(value):
        if value == -1:
            return True
        else:
            return False
    
#     print(target_value_test(value))
    
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [value],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    multiple_test(test_cases, target_value_test)
    
        
##############################################

### ex11   
def treatment_dataset_split_test(target):
    
    example_df = pd.DataFrame(columns = ['ID', 'TRTMT'])
    example_df.ID = range(100)
    example_df.TRTMT = np.random.binomial(n=1, p=0.5, size=100)
    treated_ids = set(example_df[example_df.TRTMT==1].ID)
    example_y = example_df.TRTMT.values
    
    x_treat_train, y_treat_train, x_treat_val, y_treat_val, x_control_train, y_control_train, x_control_val, y_control_val = treatment_dataset_split_test_case(target, example_df, example_y)
    
    value = True

    print("Tests:")
    pass_flag = True
    value += pass_flag
    
    pass_flag = (len(x_treat_train) + len(x_treat_val) + len(x_control_train) + len(x_control_val) == 100)
    value += pass_flag
    print(f"\nDidn't lose any subjects: {pass_flag}")
    
    pass_flag = (("TRTMT" not in x_treat_train) and ("TRTMT" not in x_treat_val) and ("TRTMT" not in x_control_train) and ("TRTMT" not in x_control_val))
    value += pass_flag
    print(f"\nTRTMT not in any splits: {pass_flag}")
    
    split_treated_ids = set(x_treat_train.ID).union(set(x_treat_val.ID))
    pass_flag = (len(split_treated_ids.union(treated_ids)) == len(treated_ids))
    value += pass_flag
    print(f"\nTreated splits have all treated patients: {pass_flag}")
    
    split_control_ids = set(x_control_train.ID).union(set(x_control_val.ID))
    pass_flag = (len(split_control_ids.intersection(treated_ids)) == 0)
    value += pass_flag
    print(f"\nAll subjects in control split are untreated: {pass_flag}")
    
    pass_flag = (len(set(x_treat_train.ID).intersection(x_treat_val.ID)) == 0)
    value += pass_flag
    print(f"\nNo overlap between treat_train and treat_val: {pass_flag}")
    
    pass_flag = (len(set(x_control_train.ID).intersection(x_control_val.ID)) == 0)
    value += pass_flag
    print(f"\nNo overlap between control_train and control_val: {pass_flag}")
    
    print(f"\n--> Expected: All statements should be True\n")
    
    def target_value_test(value):
        if value == 8:
            return True
        else:
            return False
        
    expected_output = True
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [value],
            "expected": expected_output,
            "error":"Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [value],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [value],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target_value_test)
        
def get_labels_test(target):
    
    test_sentences = test_sentences_test_case()
    
    print("Test Case:\n")
    print("Test Sentences:\n")
    for s in test_sentences:
        print(s)
    print("\n")
    
    print("Retrieved Labels:\n")
    retrieved_labels = target(test_sentences)
    for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print("\n")
    
    expected_output = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': True, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex2
def get_labels_negative_aware_test(target):
    
    test_sentences = test_sentences_test_case()
    
    print("Test Case:\n")
    print("Test Sentences:\n")
    for s in test_sentences:
        print(s)
    print("\n")
    
    print("Retrieved Labels:\n")
    retrieved_labels = target(test_sentences)
    for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print("\n")
    
    expected_output = expected_labels = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': False, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    



##############################################        
### ex3
def prepare_bert_input_test(target, tokenizer):
    
    passage, question = prepare_bert_input_test_case()
    
    print("Test Case:\n")
    print("Passage: ", passage)
    print("Question: ", question, "\n")
    
    max_seq_length = 20
    input_ids, input_mask, tokens = target(question, passage, tokenizer, max_seq_length)
    print("Tokens:")
    print(tokens)
    print("\nCorresponding input IDs:")
    print(input_ids)
    print("\nMask:")
    print(input_mask, "\n")
    
    def expected_output_values():
        input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        tokens = ['[CLS]', 'What', 'is', 'my', 'name', '?', '[SEP]', 'My', 'name', 'is', 'Bob', '.']
        input_ids = [101, 1327, 1110, 1139, 1271, 136, 102, 1422, 1271, 1110, 3162, 119, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids), 0)
        
        return input_ids, input_mask, tokens
    
    expected_output = expected_output_values()
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
        

##############################################    
### ex4
def get_span_from_scores_test(target):
    
    start_scores_1, end_scores_1, input_mask_1, start_scores_2, end_scores_2, input_mask_2 = get_span_from_scores_test_case()
    
    verbose = True
    
    print("Test Case 1: \n")
    start_1, end_1 = target(start_scores_1, end_scores_1, input_mask_1, verbose)
    print("Expected: (1, 4)")
    print("Returned: ({}, {})".format(start_1, end_1))
    expected_output_1 = (1, 4)
    
    print("\nTest Case 2: \n")
    start_2, end_2 = target(start_scores_2, end_scores_2, input_mask_2, verbose)
    print("Expected: (1, 1)")
    print("Returned: ({}, {})".format(start_2, end_2))
    print("\n")
    expected_output_2 = (1, 1)
    
    verbose = False
    test_cases = [
        {
            "name":"datatype_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [start_scores_2, end_scores_2, input_mask_2, verbose],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. Please check how you set the range of your for loops."
        }
    ]
    
    multiple_test(test_cases, target)
    
    
##############################################    
### ex5
def construct_answer_test(target):
    
    tmp_tokens_1, tmp_tokens_2 = construct_answer_test_case()
    
    print("Test Case: \n")
    print("Originals:\n")
    print(tmp_tokens_1)
    print(tmp_tokens_2)
    
    tmp_out_string_1 = target(tmp_tokens_1)
    tmp_out_string_2 = target(tmp_tokens_2)
    
    expected_output_1 = "hello how  are  you?"
    expected_output_2 = "@hellohowareyou?"
    
    print("\nProcessed Strings:\n")
    print(f"tmp_out_string_1: {tmp_out_string_1}, length {len(tmp_out_string_1)}")
    print(f"tmp_out_string_2: {tmp_out_string_2}, length {len(tmp_out_string_2)}")
    print("\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Wrong output for the first string."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_tokens_2],
            "expected": expected_output_2,
            "error": "Wrong output for the second string."
        }
    ]
    
    multiple_test(test_cases, target)

def grad_cam_test(target, im_path, mean, std, load_image_normalize, model, reference):
    
    im = grad_cam_test_case(target, im_path, mean, std, load_image_normalize, model)
    
    cam = target(model, im, 5, 'conv5_block16_concat') # Mass is class 5
    
    # Loads reference CAM to compare our implementation with.
    error = np.mean((cam - reference)**2)
    
    print("Error from reference should be less than 0.05")
    print("Your error from reference: ", error, "\n")
    
    expected_value = True
    
    ### to check output value is less than 0.05
    def target_value_test(error):
        if error < 0.05:
            return True
        else:
            return False
        
    ### to check output value is less than 0.05
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [error],
            "expected": expected_value,
            "error": "Wrong output. Your reference is greater than 0.05"
        }
    ]
    
    multiple_test(test_cases, target_value_test)
    
    return cam


### ex3
def permute_feature_test(target):
    
    example_df = permute_feature_test_case()
    
    print("Test Case\n")
    print("Original dataframe:\n")
    print(example_df, "\n")
    
    print("col1 permuted:\n")
    print(target(example_df, 'col1'), "\n")
    
    print("Average values after computing over 1000 runs:")
        
    def test_target_values(target):
        col1_values = np.zeros((3, 1000))

        np.random.seed(0) # Adding a constant seed so we can always expect the same values and evaluate correctly. 

        for i in range(1000):
            col1_values[:, i] = target(example_df, 'col1')['col1'].values
        
        return np.mean(col1_values, axis=1)
    
    expected_output = np.array([0.976, 1.03,  0.994])
        
    print("Average of col1: {}".format(test_target_values(target)))
    print("\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [target],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [target],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [target],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, test_target_values)

## ex4
def permutation_importance_test(target, cindex):
    print("Test Case\n")
    print("You check your answers on a Logistic Regression on a dataset")
    print("where y is given by a sigmoid applied to the important feature.") 
    print("The unimportant feature is random noise.")
    print("\n")
    
    example_df, example_y, example_model = permutation_importance_test_case()
    
    num_samples = 100
    example_importances = target(example_df, example_y, example_model, cindex, num_samples)
    
    ### to check the output's shape and data type
    expected_shape_type = pd.DataFrame({"important": 0.5, "unimportant": 0.0}, index=['importance'])
    expected_flag = True
    
    def target_output_value_test(example_importances):
        important = example_importances.iloc[0]['important']
        unimportant = example_importances.iloc[0]['unimportant']
        flag = np.allclose([0.5, 0.0], [important, unimportant], rtol=1e-01, atol=1e-02)
        
        return flag
    
    print("Computed Importances:")
    print(example_importances)
    print("\n")
        
    ### for datatype and shape check
    test_cases = [
        {
            "name":"datatype_check",
            "input": [example_df, example_y, example_model, cindex, num_samples],
            "expected": expected_shape_type,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [example_df, example_y, example_model, cindex, num_samples],
            "expected": expected_shape_type,
            "error": "Wrong shape."
        }
    ]
    
    multiple_test(test_cases, target)
    
    ### for checking closeness of expected values
    test_cases = [
        {
            "name":"equation_output_check",
            "input": [example_importances],
            "expected": expected_flag,
            "error": "Wrong output. Your values are not closer to expected 0.5 and 0.0"
        }
    ]
    
    multiple_test(test_cases, target_output_value_test)
    
