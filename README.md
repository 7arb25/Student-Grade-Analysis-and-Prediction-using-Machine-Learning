# Student Grade Analysis and Prediction using Machine Learning

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
2. **Ordinal Encoding:** The target variable "Grades" was ordinally encoded to reflect the inherent order (C < B < A).
3. **One-Hot Encoding:** Categorical features (for the model) were one-hot encoded using `pd.get_dummies()`.

---   

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the relationships between variables and identify potential trends. Key visualizations included:

*   **Distribution of Grades:** A count plot showed the overall distribution of grades.
*   **Impact of Factors on Grades:** Grouped bar charts visualized the relationship between various factors (e.g., parental education, tutoring, school environment) and student grads.
*   **Correlation Matrix:** A heatmap displayed the correlation between numerical features.


---

## Machine Learning Model


A Random Forest Classifier was trained to predict student grades. The model was evaluated using accuracy, classification report (precision, recall, F1-score), and a confusion matrix. Feature importance was also analyzed and visualized.

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

1.  Clone the repository: `git clone https://github.com/your-username/student-grade-analysis.git`
2.  Navigate to the project directory: `cd student-grade-analysis`
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
                          
[Creative Commons 4.0 License Share Alike](https://creativecommons.org/)

## Repository Structure
