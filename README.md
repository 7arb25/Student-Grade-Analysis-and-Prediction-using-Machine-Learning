# Student Grade Analysis and Prediction using Machine Learning

![img alt](https://github.com/7arb25/student-performance-analysis/blob/9665aef7c72dfdaac73086ef19fb6b1dce5fc326/Imgs/pic.jpg)
This project analyzes a dataset of university student grades and associated factors to understand the key influences on academic performance and build a predictive model for student grades.  The analysis is conducted for the InvestInMinds Foundation, a non-profit organization dedicated to supporting and educating students.

---

## Table of Contents

* [Project Overview](#project-overview)
*   [Dataset Description](#dataset-description)
*   [Data Preprocessing](#data-preprocessing)
*   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
*   [Machine Learning Model](#machine-learning-model)
*   [Results and Conclusions](#results-and-conclusions)
*   [Recommendations for InvestInMinds](#recommendations-for-investinminds)
*   [Repository Structure](#repository-structure)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Contributing](#contributing)
*   [License](#license)

----
                                         
## Project Overview

This project aims to:


1. Identify the key factors that influence university student grades.
2. Build a machine learning model to predict student grades based on these factors.
3. Provide actionable insights to the InvestInMinds Foundation to inform their programs and initiatives.
4. The analysis utilizes a dataset containing student demographics, academic habits, lifestyle choices, and other relevant information.  A Random Forest Classifier is used for predictive modeling, and various data visualization techniques are employed to explore the relationships between variables.

---

## Dataset Description

This document provides a description of the columns present in the "Factors Affecting University Student Grades" dataset.  The dataset aims to capture various factors that might influence a student's academic performance.

### Student Demographics and Background

*   **Age:**  The age of the student. (Numerical)
*   **Gender:** The gender of the student (Male/Female). (Categorical)
*   **Parental_Education:** The highest level of education attained by the student's parents (e.g., High School, Some College, College, Graduate). (Categorical)
*   **Family_Income:** The annual family income. (Numerical/Categorical - could be treated as categorical if ranges are used).  Contains one "Unknown" value.
*   **Previous_Grades:** The student's grades in previous academic years (e.g., A, B, C). (Categorical)

### Academic Factors

*   **Attendance:** The student's attendance rate. (Numerical - likely a percentage)
*   **Class_Participation:** The level of the student's participation in class (e.g., High, Medium, Low). (Categorical)
*   **Study_Hours:** The number of hours the student studies per week. (Numerical)
*   **Major:** The student's major or field of study (e.g., Business, Science, Engineering). (Categorical)
*   **School_Type:** The type of school the student attends (e.g., Public, Private). (Categorical)
*   **Financial_Status:** The student's financial status (e.g., Low, Medium, High). (Categorical)
*   **Parental_Involvement:** The level of parental involvement in the student's education (e.g., High, Medium, Low). (Categorical)
*   **Educational_Resources:**  Availability of educational resources for the student (Yes/No). (Categorical)
*   **Motivation:** The student's motivation level (e.g., High, Medium, Low). (Categorical)
*   **Self_Esteem:** The student's self-esteem level (e.g., High, Medium, Low). (Categorical)
*   **Stress_Levels:** The student's stress levels (e.g., High, Medium, Low). (Categorical)
*   **School_Environment:** The quality of the school environment (e.g., Positive, Negative, Neutral). (Categorical)
*   **Professor_Quality:** The perceived quality of the professors (e.g., High, Medium, Low). (Categorical)
*   **Class_Size:** The size of the classes the student attends. (Numerical)
*   **Extracurricular_Activities:** Participation in extracurricular activities (Yes/No). (Categorical)
*   **Learning_Style:** The student's preferred learning style (e.g., Visual, Auditory, Kinesthetic). (Categorical)
*   **Tutoring:** Whether the student receives tutoring (Yes/No). (Categorical)
*   **Mentoring:** Whether the student has a mentor (Yes/No). (Categorical)

### Lifestyle and Personal Factors

*   **Sleep_Patterns:** The student's sleep patterns (e.g., Adequate, Inadequate). (Categorical - this needs further clarification on the actual values)
*   **Nutrition:** The student's nutrition habits (e.g., Healthy, Unhealthy, Balanced). (Categorical)
*   **Physical_Activity:** The student's level of physical activity (e.g., High, Medium, Low). (Categorical)
*   **Screen_Time:** The amount of time the student spends on screens. (Numerical)
*   **Educational_Tech_Use:** The student's use of educational technology (Yes/No). (Categorical)
*   **Peer_Group:** The influence of the student's peer group (e.g., Positive, Negative, Neutral). (Categorical)
*   **Bullying:**  Experiences with bullying (Yes/No). (Categorical)
*   **Study_Space:** The quality of the student's study space (Yes/No). (Categorical)
*   **Lack_of_Interest:** Level of lack of interest in studies. (Numerical/Categorical - needs more detail)
*   **Time_Wasted_on_Social_Media:** Time wasted on social media. (Numerical)
*   **Sports_Participation:** Participation in sports (Yes/No). (Categorical)

### Target Variable

*   **Grades:** The student's grades (e.g., A, B, C, etc.). This is the target variable that we want to predict or analyze. (Categorical/Numerical - depending on how grades are represented)


**Note:**  Some categorical variables may need to be converted into numerical representations (e.g., one-hot encoding) before being used in machine learning models.  Also, the "Unknown" value in `Family_Income` should be handled appropriately (e.g., imputation or removal).  The specific categories within some columns (like `Sleep_Patterns`, `Lack_of_Interest`) would benefit from further clarification in the dataset documentation.                                                

---

## Data Preprocessing
                                                                     
The following preprocessing steps were performed:
1. **Missing Value Imputation:** Numerical features were imputed using KNNImputer, and categorical features were imputed using SimpleImputer (most frequent value).

#### Heatmap Before

``` python
missing_data = df.isnull()

# 4. Create the heatmap
plt.figure(figsize=(10, 8)) 
sns.heatmap(missing_data, cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()
```

![img alt](https://github.com/7arb25/student-performance-analysis/blob/74e0ec7d24ff6c15d2cb4bcafbfebbbe1c7d0d4d/Imgs/Missing%20vals%20heatmap.jpg)

``` python
from sklearn.impute import KNNImputer,SimpleImputer
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

numerical_df = df[numerical_cols]
categorical_df = df[categorical_cols]

# 3. Impute numerical features using KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)  
numerical_imputed = knn_imputer.fit_transform(numerical_df)
numerical_imputed_df = pd.DataFrame(numerical_imputed, columns=numerical_cols) 


mode_imputer = SimpleImputer(strategy='most_frequent')
categorical_imputed = mode_imputer.fit_transform(categorical_df)
categorical_imputed_df = pd.DataFrame(categorical_imputed, columns=categorical_cols)

```

#### Heatmap After

``` python

import missingno as msno

msno.matrix(df) 
plt.title('Missingno Matrix')
plt.show()
################
msno.bar(df) 
plt.title('Missingno Bar Chart')
plt.show()

```


![img](https://github.com/7arb25/student-performance-analysis/blob/9665aef7c72dfdaac73086ef19fb6b1dce5fc326/Imgs/full%20heatmap2.jpg)

2. **Ordinal Encoding:** The target variable "Grades" was ordinally encoded to reflect the inherent order (C < B < A).
3. **One-Hot Encoding:** Categorical features (for the model) were one-hot encoded using `pd.get_dummies()`.

---   

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the relationships between variables and identify potential trends. Key visualizations included:

*   **Distribution of Grades:** A count plot showed the overall distribution of grades.
*   **Impact of Factors on Grades:** Grouped bar charts visualized the relationship between various factors (e.g., parental education, tutoring, school environment) and student grads.
*   **Correlation Matrix:** A heatmap displayed the correlation between numerical features.

#### Detailed Description 

``` python
factors_to_analyze = ['Parental_Education', 'Study_Hours', 'School_Type', 'Financial_Status', 'Tutoring', 'Mentoring', 'Educational_Resources', 'School_Environment', 'Time_Wasted_on_Social_Media']

for factor in factors_to_analyze:

    if factor in categorical_cols: #Use countplot for categorical variables
      sns.countplot(x=factor, hue='Grades', data=df)
      plt.title(f'Impact of {factor} on Grades')
      plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if needed
      plt.tight_layout() # Adjust layout to prevent labels from overlapping
      plt.show()
    elif factor in numerical_cols: #Use barplot for numerical variables
      sns.barplot(x='Grades', y=factor, data=df)
      plt.title(f'Impact of {factor} on Grades')
      plt.show()

```

![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/tutoring_grads.jpg)
  - **Tutoring and Grade Distribution:**


      * Yes" Tutoring: Among students who received tutoring, the most common grade is "B," followed by "A," and then "C."
      * "No" Tutoring: Among students who did not receive tutoring, the distribution is slightly different. "B" is still the most frequent, but the number of "A" grades is notably lower, and the number of "C" grades is slightly higher compared to the "Yes" tutoring group.

  - Impact of Tutoring:
      * Increased "A" Grades: A higher proportion of students who received tutoring achieved an "A" grade compared to those who did not. This suggests that tutoring might have a positive impact on achieving higher grades.
      * Reduced "C" Grades: There's a slight decrease in the proportion of "C" grades among students who received tutoring.
      *  "B" Grades Remain High: While tutoring seems to have a positive influence on "A" grades, the number of "B" grades remains relatively high in both groups. This could indicate that other factors also play a significant role in achieving a "B" grade.
  - Potential Implications:
      * Tutoring Effectiveness: The data suggests that tutoring could be an effective intervention for improving student grades, particularly for those aiming for an "A.”
  - Program Evaluation: This data supports the idea that tutoring programs can be beneficial. InvestInMinds could use this information to evaluate and potentially expand their tutoring initiatives.
  - Targeted Support: The findings suggest that tutoring might be particularly helpful for students aiming for higher grades ("A").
  - Further Research: InvestInMinds could conduct more rigorous research (potentially including randomized controlled trials) to establish the causal impact of tutoring and to identify other factors that contribute to student success. They might also want to explore why the "B" grade is so prevalent even with tutoring, which could point to areas for additional intervention.

--- 
![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/mentoring_grads.jpg)

1. **Grade Distribution by Mentoring Status:**

    - "Yes" Mentoring: Among students who had mentors, the most frequent grade is "B," followed by "A," and then "C."
    - "No" Mentoring: The distribution is similar for students without mentors, with "B" being the most common, followed by "A," and then "C."

2. **Impact of Mentoring:**

    - **Slightly Higher "A"s with Mentoring:** A slightly higher number of students who had mentors achieved an "A" grade compared to those who didn't have mentors. This suggests a potential positive influence of mentoring on achieving higher grades.
    - **Similar "B" and "C" Trends:** The counts of "B" and "C" grades are relatively similar between the two groups (mentored vs. not mentored).

3. **Potential Implications:
Mentoring's Potential Benefit:** The data hints at a possible positive relationship between mentoring and achieving higher grades, particularly "A"s.

**In summary,** the chart suggests a possible positive association between mentoring and higher grades, but further research is needed to confirm a causal relationship and rule out the influence of other factors.  The information can be valuable for InvestInMinds in evaluating and refining their programs.

---
![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/parential_edication_grade.jpg)

1. **Grade Distribution Across Parental Education Levels:**

    - `"College":` Among students whose parents' highest education is "College," the most frequent grade is "B," followed by "A," and then "C."
    - `"Some College":` The trend is similar for students whose parents attended "Some College," with "B" being the most frequent, followed by "A," and then "C." However, the overall count in this category is the highest.
    - `"High School":` For students whose parents' highest education is "High School," the distribution is slightly different. The count of "B" grades is still the highest, but the count of "C" grades is relatively higher compared to the "College" and "Some College" categories.
    - `"Graduate":` Students whose parents are "Graduates" show a similar pattern to the "College" and "Some College" groups, with "B" being the most common, followed by "A," and then "C."

2. **Impact of Parental Education on Grades:**

    - `Higher Parental Education, Potentially Higher "A"s:` There is a tendency, though not absolute, for the proportion of "A" grades to be slightly higher in the "College," "Some College," and "Graduate" categories compared to the "High School" category. This suggests that higher parental education might be associated with a slightly increased likelihood of students achieving higher grades.
    - `"B" Grade Consistency:` The "B" grade remains the most frequent across all parental education levels.
    - `"C" Grades and Lower Parental Education:` The "High School" category shows a slightly higher proportion of "C" grades compared to the other categories, suggesting that students whose parents have a high school education might be slightly more likely to receive lower grades.

3. **Potential Implications:**

    - `Parental Education as a Factor:` The data hints at parental education being a potential factor influencing student grades.

**Targeted Interventions:** The data suggests that students whose parents have lower levels of education might benefit from additional support and resources.
**Holistic Approach:** InvestInMinds should consider a holistic approach that addresses multiple factors, recognizing that parental education is just one piece of the puzzle.

---
![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/school_type_grade.jpg)

1. **Grade Distribution Across School Types:**
   
    - `"Private":` Among students attending private schools, the most frequent grade is "B," followed by "A," and then "C."
    - `"Public":` The distribution is similar for students attending public schools, with "B" being the most common, followed by "A," and then "C."

2. **Impact of School Type on Grades:**


    - `Similar Trends:` The overall patterns of grade distribution are quite similar between private and public schools. Both show a higher concentration of "B" grades, followed by "A" and then "C."
    - `Slight Variations:` There are some minor variations in the heights of the bars, suggesting slightly different proportions of grades between the two school types. However, these differences don't appear to be substantial at first glance.

3. **Potential Implications:**

    - `School Type Might Not Be the Dominant Factor:` The data suggests that the type of school (private or public) might not be the most influential factor in determining student grades, as the distributions are relatively similar.

**In summary,** the chart suggests that school type alone might not be a strong predictor of student grades, and other factors are likely playing a more significant role.  InvestInMinds should consider this when developing strategies to support student success

---

![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/financial_status_grade.jpg)

1. **Holistic Understanding:** InvestInMinds should consider a holistic understanding of student success, recognizing that financial status is just one factor among many.
2. **Focus on Multifaceted Support:** Given the relatively consistent grade distributions, InvestInMinds might want to focus on providing multifaceted support that addresses various needs, such as academic resources, mentoring, tutoring, and socio-emotional support, rather than solely focusing on financial assistance.
3. **Further Research:** InvestInMinds could conduct further research to explore how financial status interacts with other factors to affect student grades. They might also want to investigate the specific challenges faced by students from different financial backgrounds and how these challenges can be addressed through targeted interventions.

**In summary,** the chart suggests that financial status alone might not be a strong predictor of student grades, and other factors are likely playing a more significant role.  InvestInMinds should consider this when developing strategies to support student success. They should focus on a wide range of support mechanisms and conduct further research to gain a deeper understanding of the complex factors influencing student outcomes.

---

![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/educational_resourses_grade.jpg)

1. **Grade Distribution Across Resource Availability:**

    - `"Yes" Resources:` Among students with access to educational resources, the most frequent grade is "B," followed by "A," and then "C."
    - `"No" Resources:` The trend is similar for students without access to resources, with "B" being the most common, followed by "A," and then "C."

2. **Impact of Educational Resources on Grades:**

    - `Slightly Higher "A"s with Resources:` There appears to be a slightly higher proportion of "A" grades among students who have access to educational resources compared to those who don't.
    - `Similar "B" and "C" Trends:` The counts of "B" and "C" grades are relatively similar between the two groups (resource access vs. no access).

3. **Potential Implications:**

    - `Resources Might Have a Positive Influence:` The data hints at a possible positive relationship between access to educational resources and achieving higher grades, particularly "A"s.

**Resource Provision as a Potential Strategy:** The data suggests that providing educational resources could be a beneficial strategy for improving student outcomes.
**Targeted Resource Allocation:** InvestInMinds might consider focusing on providing resources to students who lack access, particularly those who are aiming for higher grades.

---

![img](https://github.com/7arb25/student-performance-analysis/blob/982c28af3468eae0b062388b72d6a811a61118e4/Imgs/school_environment_grade.jpg)

1. **Grade Distribution Across School Environments:**

    - `"Negative":` Among students in a negative school environment, the most frequent grade is "B," followed by "A," and then "C."
    - `"Positive":` The trend is similar for students in a positive school environment, with "B" being the most common, followed by "A," and then "C."
    - `"Neutral":` Students in a neutral school environment also follow the same pattern, with "B" being the most frequent, followed by "A," and then "C."

2. **Impact of School Environment on Grades:**

    - `Relatively Consistent Patterns:` The distribution of grades across the three school environment categories is relatively consistent. All three groups show a similar pattern of having the highest count of "B" grades, followed by "A" and then "C."
    - `Potential Slight Variations:` While the overall patterns are similar, there might be slight variations in the exact proportions of each grade level within each school environment group. However, these differences don't appear to be very large at first glance.

3. **Potential Implications:**

    - `School Environment Might Not Be the Dominant Factor:` The data suggests that the school environment alone might not be the most influential factor in determining student grades, as the distributions are relatively similar across all three categories.

---

## Machine Learning Model


A Random Forest Classifier was trained to predict student grades. The model was evaluated using accuracy, classification report (precision, recall, F1-score), and a confusion matrix. Feature importance was also analyzed and visualized.

#### model coding


#### feature importance 

``` python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
X = pd.get_dummies(X, columns=X.select_dtypes(exclude=np.number).columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


feature_importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))  # Show top 10 features
plt.title('Top 10 Important Features')
plt.show()


print(importance_df.head(10))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categories, yticklabels=categories)  # Use original grade labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

```


![img](https://github.com/7arb25/student-performance-analysis/blob/2e39fb9390945833314dc1ec40b1258ecd1afc9e/Imgs/important%20features.jpg)

1. **Feature Importance Ranking:**

    - `Class_Size` is the most important feature, having the longest bar. This suggests that the size of the class has the strongest influence on the model's predictions (student grades).
    - `Attendance` is the second most important feature, indicating a strong correlation between attendance and grades.
    - `Study_Hours` is the third most important feature, which aligns with common expectations about academic success.
    - `Screen_Time` and Sleep_Patterns also appear to be significant factors.
    - `Time_Wasted_on_Social_Media` has a moderate level of importance.
Age shows a relatively lower importance compared to the top features.
    - `Stress_Levels_Low`, `Parental_Involvement_High`, and `Professor_Quality_Low` have the lowest importance among the top 10, but still contribute to the model's predictive power.

2. **Interpretation:**

    - The model suggests that factors related to academic engagement (`Class_Size`, `Attendance`, `Study_Hours`) and lifestyle (`Screen_Time`, `Sleep_Patterns`) are the most influential predictors of student grades.
    - While `Age`, `Stress Levels`, `Parental Involvement`, and `Professor Quality` make a contribution, their impact is comparatively smaller.

#### model limeitation

![img](https://github.com/7arb25/student-performance-analysis/blob/2e39fb9390945833314dc1ec40b1258ecd1afc9e/Imgs/confusion%20matrix_.jpg)

1. **Accuracy:**

    - Overall accuracy is calculated as the sum of correctly classified instances divided by the total number of instances.
    -    (87 + 595 + 59) / (87 + 441 + 55 + 125 + 595 + 68 + 89 + 494 + 59) = 741 / 2013 ≈ 0.368 or 36.8%
    -    This means the model is only accurate about 36.8% of the time, which is not very good.

2. **Precision:**

    - Precision for each class measures how many of the instances predicted as that class were actually correct.
    - Precision for A: 59 / (55 + 68 + 59) = 59 / 182 ≈ 0.324 (32.4%)
    - Precision for B: 595 / (441 + 595 + 494) = 595 / 1530 ≈ 0.389 (38.9%)
    - Precision for C: 87 / (87 + 125 + 89) = 87 / 301 ≈ 0.289 (28.9%)

3. **Recall (Sensitivity or True Positive Rate):**

    - Recall for each class measures how many of the actual instances of that class were correctly predicted.
    - Recall for A: 59 / (55 + 68 + 59) = 59 / 182 ≈ 0.324 (32.4%)
    - Recall for B: 595 / (441 + 595 + 494) = 595 / 1530 ≈ 0.389 (38.9%)
    - Recall for C: 87 / (87 + 125 + 89) = 87 / 301 ≈ 0.289 (28.9%)

4. **F1-Score:**

    - The F1-score is the harmonic mean of precision and recall. It balances both metrics.
    - F1-Score for A: (2 * 0.324 * 0.324) / (0.324 + 0.324) ≈ 0.324
    - F1-Score for B: (2 * 0.389 * 0.389) / (0.389 + 0.389) ≈ 0.389
    - F1-Score for C: (2 * 0.289 * 0.289) / (0.289 + 0.289) ≈ 0.289

---

## Results and Conclusions          
*   **Key Predictors:** The Random Forest model identified class size, attendance, study hours, screen time, and sleep patterns as some of the most important predictors of student grades.

*   **Holistic Influence:** The analysis suggests that both academic engagement and lifestyle factors play a crucial role in academic performance.

*   **Limitations:** The observational nature of the data limits the ability to draw causal conclusions. Further research is needed to establish a precise paper.

---


## Recommendations for InvestInMinds

1.  **Holistic Programs:** Develop holistic programs that address both academic and lifestyle factors.

2.  **Targeted Interventions:** Use the identified risk factors to target students in need of additional support.

3.  **Advocate for Smaller Class Sizes:** Advocate for smaller class sizes where feasible.

4.  **Invest in Resources and Mentoring:** Continue investing in resources and mentoring programs.

5.  **Further Research:** Conduct further research to establish causality and explore other potential factors.

6.  **Data-Driven Decisions:** Use data to track program effectiveness and inform decision-making.

7.  **Collaboration:** Collaborate with schools and families to create a supportive learning environment.

---

## Installation

1.  Clone the repository: `git clone https://github.com/7arb25/Student-Grade-Analysis-and-Prediction-using-Machine-Learning.git`
2.  Navigate to the project directory: `cd Student Grade Analysis and Prediction using Machine Learning`
3.  Create a virtual environment (recommended): `python3 -m venv .venv`
4.  Activate the virtual environment:
   *   macOS/Linux: `source .venv/bin/activate`
   *   Windows: `.venv\Scripts\activate`
5. Install the required packages: `pip install -r requirements.txt`
6. Download the dataset from Kaggle (or specify the source) and place it in the `data/` directory.

---

## Usage
                  
1.  Open the Jupyter notebook `student_grade_analysis.ipynb` in the `notebooks/` directory to reproduce the analysis.
2.  Alternatively, you can run the Python scripts in the `scripts/` directory (if any).
                          
---
                          
## Contributing
                          
Contributions are welcome! Please open an issue or submit a pull request.

---
                          
## License
                          
Abdelrahman G. A. Harb i developed it for the benefit of InvestInMinds Foundation Under The License Of [Creative Commons 4.0 License Share Alike](https://creativecommons.org/)
