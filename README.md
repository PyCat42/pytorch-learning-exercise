# PyTorch Learning Exercice

This repo contains some basic PyTorch exercises.

Original tutorial by Daniel Bourke: https://www.youtube.com/watch?v=Z_ikDlimN6A&t=2s

## In this repo
[src](src):
- [model.py](src\models.py) - Contains custom model classes.
- [ml_helpers.py](src\ml_helpers.py) - Various train, test and prediction helper functions.
- [plotting_helpers.py](src\plotting_helpers.py) - Functions for visualizing initial dataset and model results.
- [data_helpers.py](src\data_helpers.py) - Functions for data scanning and processing.
- [metrics_funcs](src\metrics_funcs.py) - Main metric functions used for assessing the model implemented by hand.

[tests](tests) - Contains scripts for testing implemented models and helper functions:

### Simple Linear Regression Model

- testing code: [simple_lin_reg_model](tests/simple_lin_reg_model)

| Linear Regression Model - No Predictions                   | Linear Regression Model - Predictions                   |
|------------------------------------------------------------|---------------------------------------------------------|
| ![](\tests\simple_lin_reg_model\lin_reg_model_no_pred.png) | ![](\tests\simple_lin_reg_model\lin_reg_model_pred.png) |


### Binary and Multiclass Circle classification

- testing code: [circles_binary_classification](tests/circles_binary_classification) 
and [circles_multiclass](tests/circles_multiclass)

| Binary Classification                                                        | Multiclass Classification                                  |
|------------------------------------------------------------------------------|------------------------------------------------------------|
| ![](tests/circles_binary_classification/example_circles_bin_class_model.png) | ![](tests/circles_multiclass/linear_decision_boundary.png) |

### FashionMNIST Classification

- testing code: [fashionMNIST_classification](tests/fashionMNIST_classification)

![Model Comparisson](tests/fashionMNIST_classification/model_comparison.png)

![Example CNN Model Prediction](tests/fashionMNIST_classification/example_prediction.png)

### Food Classification

- testing code: [food_classification](tests/food_classification)

![Model Comparison](tests/food_classification/model_curves_comparison.png)

![Custom Prediction (using model trained on augmented data)](tests/food_classification/example_custom_prediction.png)


## Requirements

All requirements are listed in [requirements.txt](requirements.txt).

Additionally, test scripts are implemented as Jupyter Notebooks so make sure you can run them.
