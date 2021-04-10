#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join(os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        print("TRAINING MODELS")
        data_dir = os.path.join("data","cs-train")
        model_train(data_dir,test=False)
        self.assertTrue(os.path.exists(os.path.join("models")))

    def test_02_load(self):
        ## load the model
        print("LOADING MODELS")
        all_data, all_models = model_load()
        print("... models loaded: ",",".join(all_models.keys()))
        country='all'
        model = all_models[country]
        self.assertTrue('fit' in dir(model))
        
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        all_data, all_models = model_load()
        print("... models loaded: ",",".join(all_models.keys()))

        ## ensure that a list can be passed
        ## test predict
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        print(result)


        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] > 0.0)



### Run the tests
if __name__ == '__main__':
    unittest.main()
