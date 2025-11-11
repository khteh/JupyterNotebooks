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

def fraction_rows_missing_test_case():
    df_test = pd.DataFrame({'a':[None, 1, 1, None], 'b':[1, None, 0, 1]})
    
    return df_test

def naive_estimator_test_case():
    sample_df_1 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_1.Time = [5, 10, 15]
    sample_df_1.Event = [0, 1, 0]
    
    sample_df_2 = pd.DataFrame({'Time': [5,5,10],
                                'Event': [0,1,0]
                               })
    
    return sample_df_1, sample_df_2

### ex3
def HomemadeKM_test_case():
    sample_df_1 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_1.Time = [5, 10, 15]
    sample_df_1.Event = [0, 1, 0]
    
    sample_df_2 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_2.loc[:, "Time"] = [2, 15, 12, 10, 20]
    sample_df_2.loc[:, "Event"] = [0, 0, 1, 1, 1]
    
    return sample_df_1, sample_df_2