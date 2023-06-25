import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#dataset = pd.read_csv('salaries_clean.csv')
#print(dataset.columns)

dataset = pd.read_csv('Levels_Fyi_Salary_Data.csv')
#print(dataset.columns)

# Cleaning up the dataset by removing outliers
#salaryDataset = dataset[['employer_name', 'annual_base_pay', 'total_experience_years', 'employer_experience_years']]
#salaryDatasetClean = salaryDataset.dropna()
#salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['annual_base_pay'] <= 1000000]
#salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['annual_base_pay'] >= 10000]
#salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['total_experience_years'] <= 25]

salaryDataset = dataset[['company', 'yearsofexperience', 'yearsatcompany', 'totalyearlycompensation']]
salaryDatasetClean = salaryDataset.dropna()
salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['totalyearlycompensation'] <= 1000000]
salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['totalyearlycompensation'] >= 10000]
salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['yearsofexperience'] <= 40]
salaryDatasetClean = salaryDatasetClean[salaryDatasetClean['yearsatcompany'] <= 30]

#print(salaryDatasetClean)

# Extract the features and target variables
#x = salaryDatasetClean[['total_experience_years', 'employer_experience_years']]
#y = salaryDatasetClean['annual_base_pay']

x = salaryDatasetClean[['yearsofexperience', 'yearsatcompany']]
y = salaryDatasetClean['totalyearlycompensation']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# Create an instance of the LinearRegression model
linearModel = LinearRegression()

# Fit the model to the training data
linearModel.fit(x_train, y_train)

# Evaluate the LinearRegression model 
y_train_pred_lr = linearModel.predict(x_train)

train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
train_rmse_lr = np.sqrt(train_mse_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)

y_test_pred_lr = linearModel.predict(x_test)

test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
test_rmse_lr = np.sqrt(test_mse_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

#print("LinearRegression:")
#print("Training Set:")
#print("Mean Squared Error (MSE):", train_mse_lr)
#print("Root Mean Squared Error (RMSE):", train_rmse_lr)
#print("R-squared Score:", train_r2_lr)
#print()
#print("Test Set:")
#print("Mean Squared Error (MSE):", test_mse_lr)
#print("Root Mean Squared Error (RMSE):", test_rmse_lr)
#print("R-squared Score:", test_r2_lr)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the training data
ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], y_train, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Training Data')

# Create a grid of values within the input range
x_range = np.linspace(x_train.iloc[:, 0].min(), x_train.iloc[:, 0].max(), 10)
y_range = np.linspace(x_train.iloc[:, 1].min(), x_train.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = linearModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a plane
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Plane')
#plt.show()
fig.savefig('training_data_lr.png')

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the test data
ax.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Test Data')

# Create a grid of values within the input range
x_range = np.linspace(x_test.iloc[:, 0].min(), x_test.iloc[:, 0].max(), 10)
y_range = np.linspace(x_test.iloc[:, 1].min(), x_test.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = linearModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a plane
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Plane')
#plt.show()
fig.savefig('test_data_lr.png')

# Create an instance of the DecisionTreeRegressor model
decisionTreeModel = DecisionTreeRegressor()

# Fit the model to the training data
decisionTreeModel.fit(x_train, y_train)

# Evaluate the DecisionTreeRegressor model
y_train_pred_dt = decisionTreeModel.predict(x_train)

train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
train_rmse_dt = np.sqrt(train_mse_dt)
train_r2_dt = r2_score(y_train, y_train_pred_dt)

y_test_pred_dt = decisionTreeModel.predict(x_test)

test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)
test_rmse_dt = np.sqrt(test_mse_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)

#print("DecisionTree:")
#print("Training Set:")
#print("Mean Squared Error (MSE):", train_mse_dt)
#print("Root Mean Squared Error (RMSE):", train_rmse_dt)
#print("R-squared Score:", train_r2_dt)
#print()
#print("Test Set:")
#print("Mean Squared Error (MSE):", test_mse_dt)
#print("Root Mean Squared Error (RMSE):", test_rmse_dt)
#print("R-squared Score:", test_r2_dt)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trainging data
ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], y_train, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Training data')

# Create a grid of values within the input range
x_range = np.linspace(x_train.iloc[:, 0].min(), x_train.iloc[:, 0].max(), 10)
y_range = np.linspace(x_train.iloc[:, 1].min(), x_train.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = decisionTreeModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a surface
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Surface')
#plt.show()
fig.savefig('training_data_dt.png')

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the test data
ax.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Test Data')

# Create a grid of values within the input range
x_range = np.linspace(x_test.iloc[:, 0].min(), x_test.iloc[:, 0].max(), 10)
y_range = np.linspace(x_test.iloc[:, 1].min(), x_test.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = decisionTreeModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a surface
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Surface')
#plt.show()
fig.savefig('test_data_dt.png')

# Create an instance of the RandomForestRegressor model
randomForestModel = RandomForestRegressor()

# Fit the model to the training data
randomForestModel.fit(x_train, y_train)

# Evaluate the RandomForestRegressor model
y_train_pred_rf = randomForestModel.predict(x_train)

train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
train_rmse_rf = np.sqrt(train_mse_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)

y_test_pred_rf = randomForestModel.predict(x_test)

test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
test_rmse_rf = np.sqrt(test_mse_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

#print("RandomForest:")
#print("Training Set:")
#print("Mean Squared Error (MSE):", train_mse_rf)
#print("Root Mean Squared Error (RMSE):", train_rmse_rf)
#print("R-squared Score:", train_r2_rf)
#print()
#print("Test Set:")
#print("Mean Squared Error (MSE):", test_mse_rf)
#print("Root Mean Squared Error (RMSE):", test_rmse_rf)
#print("R-squared Score:", test_r2_rf)

# Create a 3D figure for the training data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the training data
ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], y_train, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Training Data')

# Create a grid of values within the input range
x_range = np.linspace(x_train.iloc[:, 0].min(), x_train.iloc[:, 0].max(), 10)
y_range = np.linspace(x_train.iloc[:, 1].min(), x_train.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = randomForestModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a surface
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Surface')
#plt.show()
fig.savefig('training_data_rf.png')

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the test data
ax.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test, alpha=0.2, color='blue', label='Actual')
ax.set_xlabel('Total Experience (years)')
ax.set_ylabel('Employer Experience (years)')
ax.set_zlabel('Salary (milions)')
ax.set_title('Test Data')

# Create a grid of values within the input range
x_range = np.linspace(x_test.iloc[:, 0].min(), x_test.iloc[:, 0].max(), 10)
y_range = np.linspace(x_test.iloc[:, 1].min(), x_test.iloc[:, 1].max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Predict the output for the grid of values
z_grid = randomForestModel.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)

# Plot the predicted values as a surface
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='red', label='Predicted Surface')
#plt.show()
fig.savefig('test_data_rf.png')

st.set_page_config(layout="wide")

# Streamlit app
st.title("Salary Prediction")

# Display the predicted salaries in three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Linear Regression Model")

    st.subheader("Model Evaluation")
    st.write("Training Set:")
    st.write("Mean Squared Error (MSE):", train_mse_lr)
    st.write("Root Mean Squared Error (RMSE):", train_rmse_lr)
    st.write("R-squared Score:", train_r2_lr)
    st.write("Test Set:")
    st.write("Mean Squared Error (MSE):", test_mse_lr)
    st.write("Root Mean Squared Error (RMSE):", test_rmse_lr)
    st.write("R-squared Score:", test_r2_lr)

    st.image('training_data_lr.png')
    st.image('test_data_lr.png')


with col2:
    st.header("Decision Tree Model")

    st.subheader("Model Evaluation")
    st.write("Training Set:")
    st.write("Mean Squared Error (MSE):", train_mse_dt)
    st.write("Root Mean Squared Error (RMSE):", train_rmse_dt)
    st.write("R-squared Score:", train_r2_dt)
    st.write("Test Set:")
    st.write("Mean Squared Error (MSE):", test_mse_dt)
    st.write("Root Mean Squared Error (RMSE):", test_rmse_dt)
    st.write("R-squared Score:", test_r2_dt)

    st.image('training_data_dt.png')
    st.image('test_data_dt.png')

with col3:
    st.header("Random Forest Model")

    st.subheader("Model Evaluation")
    st.write("Training Set:")
    st.write("Mean Squared Error (MSE):", train_mse_rf)
    st.write("Root Mean Squared Error (RMSE):", train_rmse_rf)
    st.write("R-squared Score:", train_r2_rf)
    st.write("Test Set:")
    st.write("Mean Squared Error (MSE):", test_mse_rf)
    st.write("Root Mean Squared Error (RMSE):", test_rmse_rf)
    st.write("R-squared Score:", test_r2_rf)

    st.image('training_data_rf.png')
    st.image('test_data_rf.png')

# Ask the user for input
years_of_experience = st.number_input("Enter your years of experience:")
years_at_company = st.number_input("Enter your years at the current company:")

# Create a DataFrame with the user's input
#user_data = pd.DataFrame([[years_of_experience, years_at_company]], columns=['total_experience_years', 'employer_experience_years'])
user_data = pd.DataFrame([[years_of_experience, years_at_company]], columns=['yearsofexperience', 'yearsatcompany'])

# Predict the salary using the trained models
predicted_salary_lr = linearModel.predict(user_data)
predicted_salary_dt = decisionTreeModel.predict(user_data)
predicted_salary_rf = randomForestModel.predict(user_data)

with col1:
    st.subheader("Predicted Salary:")
    st.write(predicted_salary_lr)

with col2:
    st.subheader("Predicted Salary:")
    st.write(predicted_salary_dt)

with col3:
    st.subheader("Predicted Salary:")
    st.write(predicted_salary_rf)