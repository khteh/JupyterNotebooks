import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(3)

### ex1
        
def proportion_treated_test_case():

    example_df_1 = pd.DataFrame(data =[[0, 0],
                             [1, 1], 
                             [1, 1],
                             [1, 1]], columns = ['outcome', 'TRTMT'])  
    example_df_2 = pd.DataFrame(data =[[0, 0],
                             [1, 0], 
                             [0, 0],
                             [0, 0]], columns = ['outcome', 'TRTMT'])
    example_df_3 = pd.DataFrame(data =[[0, 0],
                             [1, 1], 
                             [1, 0],
                             [1, 0]], columns = ['outcome', 'TRTMT'])

    return example_df_1, example_df_2, example_df_3


### ex2
def event_rate_test_case():
    return pd.DataFrame(data =[[0, 1],
                                 [1, 1], 
                                 [1, 1],
                                 [0, 1],
                                 [1, 0],
                                 [1, 0],
                                 [1, 0],
                                 [0, 0]], columns = ['outcome', 'TRTMT'])
    
    
### ex4

def OR_to_ARR_test_case():
    
    test_p_1, test_OR_1 = (0.75, 0.5) 
    test_p_2, test_OR_2 = (0.04, 1.2)
    
    return test_p_1, test_OR_1, test_p_2, test_OR_2 


### ex 11

def treatment_dataset_split_test_case(target, example_df, example_y):
        # Tests

        example_train, example_val, example_y_train, example_y_val = train_test_split(
            example_df, example_y, test_size = 0.25, random_state=0
        )

        (x_treat_train, y_treat_train,
         x_treat_val, y_treat_val,
         x_control_train, y_control_train,
         x_control_val, y_control_val) = target(example_train, example_y_train,
                                                             example_val, example_y_val)

        return x_treat_train, y_treat_train, x_treat_val, y_treat_val, x_control_train, y_control_train, x_control_val, y_control_val
    