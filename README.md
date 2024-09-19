<h1> Project - 3 </h1>
<h3>AI Product/Service Business & Financial Modelling</h3>
# Lung Cancer Analysis & Prediction System

## 1. Data Loading
- Loaded the dataset using `pandas.read_csv()` method.
- [Code](#link-to-code)

## 2. Basic Exploration
- **Shape and Preview:** Printed the shape of the dataset and displayed the first few records using `data.head()`.
- **Information:** Displayed dataset information using `data.info()` to check data types and null values.
- **Summary:** Generated summary statistics of numerical and categorical columns using `data.describe()`.

## 3. Handling Missing and Duplicate Values
- **Missing Values:** Checked for missing values and confirmed no null values exist.
- **Duplicates:** Checked for duplicate entries and removed them using `data.drop_duplicates()`.

## 4. Data Cleaning
- Converted numerical values (1 and 2) into categorical values ("Yes" and "No") for better visualization.
- Converted gender labels from "M" and "F" to "Male" and "Female".

## 5. Custom Palette for Visualization
- Created a custom color palette for visualizations using `seaborn` styles.

## 6. Positive Lung Cancer Cases
- Filtered the dataset to include only cases where `LUNG_CANCER` is "YES" and created a new dataframe (`data_temp_pos`).

## 7. Visualizing Positive Cases
- **Age Distribution:** Used `seaborn.histplot()` to plot age distribution of positive cases, separated by gender.
- **Gender Distribution:** Used `plt.pie()` to visualize the proportion of positive cases between males and females.

## 8. Gender-wise Analysis
- **Smoking & Alcohol Habits:** Visualized gender-wise positive cases based on smoking and alcohol consumption using `seaborn.countplot()`.
- **Symptoms:** Visualized gender-wise positive cases based on symptoms like yellow fingers, anxiety, chest pain, coughing, etc.

## 9. Correlation Heatmap
- Converted categorical variables into numerical ones using `LabelEncoder` for `LUNG_CANCER` and `OneHotEncoder` for `GENDER`.
- Created a heatmap of correlations between features using `seaborn.heatmap()`.

## 10. Preprocessing for Classification
- Split the dataset into features (`X`) and target (`y`), then scaled the features using `StandardScaler`.
- Split the data into training and testing sets using `train_test_split()`.

## 11. Logistic Regression Model
- Trained a Logistic Regression model.
- **Evaluation Metrics:** Accuracy score, confusion matrix, classification report.

## 12. Gaussian Naive Bayes Model
- Trained a Gaussian Naive Bayes model.
- **Evaluation Metrics:** Accuracy score, confusion matrix, classification report.

## 13. Bernoulli Naive Bayes Model
- Trained a Bernoulli Naive Bayes model.
- **Evaluation Metrics:** Accuracy score, confusion matrix, classification report.

## Results and Insights
- Summary of findings and comparison between models.
