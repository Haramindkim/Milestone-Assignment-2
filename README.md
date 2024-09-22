# Milestone 2 Assignment

This Python script demonstrates the use of Principal Component Analysis (PCA) for dimensionality reduction and Logistic Regression for classification on the Anderson Cancer center dataset.

## Features

- Loads the Anderson Cancer center dataset
- Applies PCA to the dataset
- Visualizes the cumulative explained variance
- Reduces the dataset to 2 principal components
- Trains a Logistic Regression model on the reduced dataset
- Evaluates the model's accuracy

## Requirements

- Python 3.x
- scikit-learn
- matplotlib
- numpy

## Usage

1. Ensure you have all required libraries installed:

```
pip install scikit-learn matplotlib numpy
```

2. Run the script:
```
python main.py
```

3. The script will output:
- Variance explanation ratios for each principal component
- A plot of cumulative explained variance
- The shape of the reduced dataset
- The accuracy of the Logistic Regression model

## Output

- Console output showing variance ratios and model accuracy
- A matplotlib graph displaying cumulative explained variance
