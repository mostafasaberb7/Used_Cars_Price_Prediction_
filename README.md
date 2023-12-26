## Car Price Estimation Model

### Business Problem Documentation

#### Problem Statement

The current process of selling a car involves taking the car to a respective company workshop or making an appointment for the company to provide an estimate of the price. This traditional process consumes a significant amount of time and resources for both the customer and the company. The objective is to develop a model for third-party companies that can provide a quick estimate of the car's price directly through their online portal. This will not only save time for customers but also help companies reduce costs and streamline the process of selling used cars.

#### Solution Outline

##### 1. Data Cleaning

###### 1.1 Investigation

- **Numerical Values Mixed with Text:**
  - Columns (Mileage, Engine, Power) contain a mix of numerical values and text.
  - In the Mileage column, there are two units present: kmpl and km/kg.

- **New_Price Column:**
  - A significant portion of the New_Price column consists of null values.
  - Two possible solutions:
    - Remove the column entirely.
    - Explore options to impute or fill in some data.

- **Null Values in Other Columns:**
  - Null values are present in other columns.

- **Seats Column:**
  - The Seats column contains some values that are zero.

- **Duplicated Rows:**
  - Duplicated rows have been identified in the dataset.

- **Power Column:**
  - The Power column includes values like "null bhp."

###### 1.2 Working with Data Issues

- Drop "New_Price" column due to a large number of null values.
- Drop rows with missing values (NaN) from the DataFrame.
- Handle specific issues in the "Power" column, where some values are "null bhp."
- Convert units in columns (Mileage, Engine, Power) to numerical values.

##### 2. Feature Engineering

- Extract relevant information from the "Name" column to create a new "Brand" column.
- Simplify the "Name" column by keeping only the first two words.
- Create additional features or transformations based on data analysis and business understanding.

##### 3. Data Analysis

- Explore value counts for key columns such as Brand, Name, Location, Fuel_Type, Transmission, etc.
- Answer specific business questions through data analysis.
- Visualize insights using interactive plots.

##### 4. Data Understanding

###### 4.1 Univariate Analysis

- Examine the distribution of numerical columns and identify outliers.
- Visualize the distribution of categorical columns.

###### 4.2 Bivariate Analysis

- Explore relationships between variables.
- Check correlations and patterns.

##### 5. Data Pre-processing

###### 5.1 Data Splitting

- Split the dataset into training and testing sets to avoid data leakage.

###### 5.2 Data Transformation

####### 5.2.1 Numerical Transformation

- Use techniques like scaling (RobustScaler) to transform numerical features.

####### 5.2.2 Categorical Transformation

- Transform ordinal data (Owner_Type) using mapping.
- Implement binary encoding for nominal data (Name, Location, Fuel_Type, Transmission, Power, Brand).

##### 6. Modeling

###### 6.1 Linear Regression

- Apply linear regression for price estimation.
- Evaluate the model's performance using R-squared score.

#### Required Packages

```bash
pip install numpy pandas scikit-learn plotly category-encoders datasist
```

#### Usage Example

```python
# Read the dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Apply the data cleaning and preprocessing steps as outlined

# Split the data
X, y = df.drop("Price", axis=1), df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Apply data transformations
# ...

# Train the linear regression model
reg = LinearRegression().fit(X_train, np.log(y_train))

# Evaluate the model
test_score = reg.score(X_test, np.log(y_test))
train_score = reg.score(X_train, np.log(y_train))

# Print the scores
print(f"Test R-squared Score: {test_score:.4f}")
print(f"Train R-squared Score: {train_score:.4f}")
```

Note: This example assumes the completion of data cleaning, feature engineering, and preprocessing steps. Adjustments may be needed based on further analysis and model performance.