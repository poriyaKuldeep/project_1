import os
import sys
import pandas as pd
import numpy as np
# import dill
import pickle
from sklearn.model_selection import cross_val_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))) :
            model=list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            cross_val=cross_val_score(model,X_train,y_train,cv=5,scoring="accuracy").mean()
            report[list(models.keys())[i]] = cross_val

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    


                          