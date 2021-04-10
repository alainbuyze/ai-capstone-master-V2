#!/usr/bin/env python
"""
api tests

"""

import sys
import os
import unittest
import requests
import re
import json
from ast import literal_eval
import numpy as np

port = 8080

try:
    requests.post('http://0.0.0.0:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    print(" Warning - API is not responding at http://0.0.0.0:8080, was the app started?\n No tests were run")
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
      
        request_json = {'mode':'test'}
        r = requests.post('http://0.0.0.0:{}/train'.format(port), json=request_json)
        train_complete = re.sub("\W+", "", r.text)
        self.assertEqual(train_complete, 'true')
    
    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        # ensure appropriate failure types
        """
    
        ## provide no data at all 
        r = requests.post('http://0.0.0.0:{}/predict'.format(port))
        print('*** test2 ',r.text)
        self.assertEqual(re.sub('\n|"', '', r.text), "[]")

        ## provide improperly formatted data
        r = requests.post('http://0.0.0.0:{}/predict'.format(port), json={"key":"value"})
        self.assertEqual(re.sub('\n|"', '', r.text),"[]")
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_03_predict(self):
        """
        # test the predict functionality
        """
 
        '''
         query_data = {'country': 'all',
        '              'year': '2018',
        '              'month': '01',
        '              'day': '05'
        ''}
        '''
        query_data = {'countries': ['all', 'france', 'netherlands', 'united_kingdom'],
                  'dates': ['01-May-2018', '02-May-2018', '01-May-2018', '01-May-2018']}
        query_type = 'dict'
        request_json = {'query':query_data, 'type':query_type, 'mode':'test'}

        r = requests.post('http://0.0.0.0:{}/predict'.format(port), json=request_json)
        print("\n",r.text)
        response =  json.loads(r.text)
        print(response)
        self.assertTrue(response['y_pred'][0] > 0.0)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_04_logs(self):
        """
        # test the log functionality
        """

        file_name = 'train-test.log'
        request_json = {'file':'train-test.log'}
        r = requests.get('http://0.0.0.0:{}/logs/{}'.format(port, file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_05_ingest(self):
        """
        test the ingest functionality
        """

        request_json = {'mode': 'test'}
        r = requests.post('http://0.0.0.0:{}/ingest'.format(port), json=request_json)
        print(r)
        train_complete = re.sub("\W+", "", r.text)
        self.assertEqual(train_complete, 'true')


### Run the tests
if __name__ == '__main__':
    print("*** Start Unit Test ***")
    unittest.main()
