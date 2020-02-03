# Model Performance Investigator

## Short Description
        Model Performance Investigator is used to analyse the performance Machine Learning and
    Deep learning models. It gives the user basic idea how the model performance on given data.
    So that user can start work on the right model.
    
## Installation
    pip install model-performance-analyzer
    
## How to use ?
    
    * Import the package.
        from model_analyzer import ml_models as ml
    
    * Call the predector function, assign it to a variable and run.
        data = [train_X, train_y, text_x]
        new_out = ml.predector('regression', data, alg_type=['linear','lasso'], 
                  score_type=['r2'], tune_param='default')

## Parameters ?
    1. 'regression' -> Current Version uses only regression type problems. It is mandatory.
    
    2. data -> data must contain "train_X, train_y, text_x". train_X, train_y is for fitting
               the model and test_x is for predicting. It is mandatory.
               
    3. alg_type -> Type of the algorithm. It is optional parameter. Provide the parameter
                   values in list.  
                   Default is "Linear Regression" algorithm.
                   
                   Currently it Supports:
                   
                    Algorithm Name             | Parameter 
                    ---------------------------|----------
                    Linear Regression        | linear
                    Polynomial Regression    | polynomial
                    Redig Regression         | ridge
                    Lasso Regression         | lasso
                    ElasticNet Regression    | elasticnet
                    Random Forest Regression | random_forest_regressor
                    XG Boost Regressor       | xgb_regressor
                    
    4. score_type -> Type of the score. It is optional parameter. Provide the parameter
                     values in list.   
                     Default is "r2" algorithm.
                   
                   Currently it Supports:
                   * r2
                   * explained_variance
                   * max_error
                   * neg_mean_absolute_error
                   * neg_mean_squared_error
                   * neg_mean_squared_log_error
                   * neg_median_absolute_error
                   * neg_mean_poisson_deviance
                   * neg_mean_gamma_deviance
            
    5. tune_param -> Tuning Parameter for the model. It is optional. By Default parameter 
                     this package uses some basic parameters for each models.
                     User can provide own Tuning Parameter.
                     
                     Template:
                         tune_param = {Name of the model: {parameters}}
                         Name of the model -> should be similar like alg_type.
                         
                     Example:
                         linear_tune_param = {
                              "fit_intercept"       : [True, False],
                              "normalize"           : [True, False],
                              "copy_X"              : [True, False],
                              "n_jobs"              : [ -1]
                            }
                        lasso_tune_param = {
                              "alpha":[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 
                                       1, 5, 10, 15, 20, 40, 50,
                                       85, 100, 300, 500, 1000
                                       ]
                            }
                            
                         tune_param = {'linear': linear_tune_param, 
                                      'lasso': lasso_tune_param}
                     
                     Note : User don't have to specify tuning parameter for all the
                            models used in 'alg_type'. If tuning parameter is not provided
                            then this package use default tuning parameter.
                            
## Project History
   The project was started in 2019 by Srikandan Rajua and Sathish Anandha.