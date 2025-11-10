import numpy as np
import pandas as pd
np.random.seed(3)

### ex1
def make_standard_normal_test_case():
    tmp_train = pd.DataFrame({'field1': [1,2,10], 'field2': [4,5,11]})
    tmp_test = pd.DataFrame({'field1': [1,3,10], 'field2': [4,6,11]})
    
    return tmp_train, tmp_test

### ex3
def cindex_test_case():
    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    scores_1 = np.array([0, 1, 1, 0])
    scores_2 = np.array([1, 0, 0, 1])
    scores_3 = np.array([0.5, 0.5, 0.0, 1.0])
    
    return y_true, scores_1, scores_2, scores_3 

