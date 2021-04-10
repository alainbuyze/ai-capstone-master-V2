# IBM AI Enterprise Workflow Capstone - Alain Buyze
Dear Evaluator, 
please find described how you can verify how I fulfilled the different evaluation criteria for this Capstone project. You should normally be able to run all the below steps in a Ubuntu environment wwith Python 3.8 and with all the Git code downloaded on your local machine in a seperate directory with the subdirectory structure preserved.

### Are there unit tests for the API?
The unittests are available in the directory **unittests**, the python module **ApiTests.py**

#### First start the FLASK API
Set the current directory to the root of the local replica of the repository

Run the command **python app.py**

Check if the app is runnin by opening **http://0.0.0.0:8080/** - you should seee the welcome page

#### Run the tests
From the root directory of the repository, run **python unittests/ApiTests.py**

This will execute 5 test:
1. test the train functionality
2. test for appropriate failure in case of no data or ill-formed input
3. test the predict functionality
4. test the log functionality
5. test the data ingestion (invoices to ts data regeneration)

## Are there unit tests for the model?
The unittests are available in the directory **unittests**, the python module **ModelTests.py**

From the root directory of the repository, run **python unittests/ModelTests.py**

This will execute 3 test:
1. Test the train functionality
2. test if models have been loaded
3. test the predict functionality

## Are there unit tests for the logging?
The unittests are available in the directory **unittests**, the python module **LoggerTests.py**

From the root directory of the repository, run **python unittests/LoggerTests.py**

This will execute 4 test:
1. Test if training log file is created
2. Test if content of Training log file can be retrieved 
3. Test if predict log file is created
4. Test if content of Preduct log file can be retrieved

## Can all of the unit tests be run with a single script and do all of the unit tests pass?
From the root directory of the repository, run **python runtests.py** 
 
## Is there a mechanism to monitor performance?
The Python Module **monitoring.py** takes care of that. 

It loads the latest train data for all countries (**latest-train0_1.pickle**) and des an outlier analysis using **wasserstein** 

## Was there an attempt to isolate the read/write unit tests from production models and logs?
This is achieved by the **parameter test=True** in calling the module **update_train_log** and **update_predict_log** in the **LoggerTests.py** module. 

This will ensure that the logfiles names have no date identifier in their name, unlike the regular log files. 


## Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?
Please check the **App.py** the "predict" routine takes a dataframe than can contain multiple countries/dates which are calculated by the "xmodel_predict" routine in the **Model.py** module. This is tested in the **unittests/ApiTests.py**, test3. 

## Does the data ingestion exists as a function or script to facilitate automation?
The data ingesten is coded as an api in **App.py** , @app.route('/ingest', methods=['GET','POST']). This will regenerate the TS data from the original invoice data from scratch. This is tested through test case ***test_05_ingest*** in **unittests/ApiTests.py**. 


## Were multiple models compared?
Check out the Python Notebook **CapstoneModelComparison.ipynb** to see the different models being compared

Models compared are: 
1. LinearRegression
2. MLPRegressor
3. KNeighborsRegressor
4. RandomForestRegressor
5. SVR

Based upon this the model RandomForestRegressor() is kept and analysed for different parameters
1. 'n_estimators': [20, 50, 100],
2. 'max_features': ['auto', 'sqrt', 'log2'],
3. 'max_depth' : [range(5,15)]


## Did the EDA investigation use visualizations?
Check out the Python Notebook **Capstone EDA.ipynb** to see the visualisations used
1. -> graphically the views per month
2. -> revenue per month
3. -> evolution of the Revenue
4. -> view spectral relationship between revenue and date 
5. -> boxplot of revenue, unique streams, total views per month and per year
6. -> correlation pairplot

## Is everything containerized within a working Docker image?
Open a terminal from the root directory of the repository and run the commands
1. docker build -t capstone-revenue-ml .
2. docker image ls (to see the image **capstone-revenue-ml** was created
3. docker run -p 4000:8080 capstone-revenue-ml
4. check that the Flask API is running at http://0.0.0.0:8080/dashboard


## Did they use a visualization to compare their model to the baseline model?
Check out the Python Notebook **CapstoneModelComparison.ipynb** the last part of it chaptrer **Comparing the model with the actual data and with the baseline model**.
I created a dataframe with the real y-values, the baseline model y-values and my model y-values and summarized by month and plottted a bar chart to compare the results.
