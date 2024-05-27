# Prediction of Occupancy Levels and optimising real estate asset management strategies based on key metrics #
# Code & Resources Used #
1. Python Version: 3.13
2. Packages: Pandas, Scikit-learn, Numpy, Matplotlib, seaborn, pandas
3. Deployment :FastApi

# Table of Contents
[Project Overview](#project-overview)

[Analysis](#analysis)

[Machine Learning Model](#machine-learning-model)

[Conclusion](#conclusion)

[Optimization strategies and recommendations for improving occupancy rates and maximising property values ](#Optimization-strategies-and-recommendations-for-improving-occupancy-rates-and-maximising-property-values.)

# Project Overview
The objective of this analysis is to evaluate and optimise real estate asset management strategies based on key metrics such as occupancy rates, square footage, and property values. By examining these metrics across different properties, asset managers can identify trends, areas for improvement, and opportunities to enhance occupancy levels and property values, ultimately maximising returns on investment and portfolio performance.
# Analysis
## Univariate Analysis


## Occupancy Analysis
Here we will evaluate the following:

●	Analyse average occupancy rates across different properties.

●	Identify properties with high and low occupancy rates.

●	Assess trends in occupancy rates over time and by property type.

## Square Footage Analysis
Under this analysis we will evaluate the following:

●	Evaluate the distribution of gross leasable area (GLA) across properties.

●	Analyse the ratio of occupied square footage to total GLA.

●	Identify opportunities to optimise space utilisation and maximise occupancy.

## Property Value Assessment
Under this section we will evaluate the following:

●	Evaluate the distribution of gross leasable area (GLA) across properties.

●	Analyse the ratio of occupied square footage to total GLA.

●	Identify opportunities to optimise space utilisation and maximise occupancy.

# Machine Learning Model

Three models were used and evaulted them with using accuracy score.

Model Listing :

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier

## Model Perfomance

Decision Tree Classifier: Training set score: 1.0000
                          Test set score: 1.0000
                          
Random Forest Classifier: Model accuracy score with 100 decision-trees : 1.0000

Logistic Regression     : Accuracy on training data :  0.9974979149291076
                        : Accuracy on test data :  1.0

# Deployment
This is a client-facing API using FastApi run into cmd

# Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/Onsombi-Felisha/Real-Estate--Asset-Management.git
    ```
2. Install dependencies:
    ```sh
    cd your-repo
    pip install -r requirements.txt
    ```

3. Usage

Run the application using:
```sh
uvicorn main:app --reload
```

4. Accessing the API: Once the API server is running, users can access it using a web browser or tools like Postman. By default, the API will be accessible at http://localhost:8000.

5. Making Predictions: Users can make predictions by sending HTTP requests to the API endpoints defined in the FastAPI application.
