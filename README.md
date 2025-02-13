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

The dataset used in this project is "Factors Affecting University Student Grades," available on Kaggle (or specify the source if different). It contains the following columns:                                
*   **Student Demographics and Background:** Age, Gender, Parental\_Education, Family\_Income

*   **Academic Factors:** Previous\_Grades, Attendance, Class\_Participation, Study\_Hours, Major, School\_Type, Financial\_Status, Parental\_Involvement, Educational\_Resources, Motivation, Self\_Esteem, Stress\_Levels, School\_Environment, Professor\_Quality, Class\_Size, Extracurricular\_Activities, Learning\_Style, Tutoring, Mentoring
                                 
*   **Lifestyle and Personal Factors:** Sleep\_Patterns, Nutrition, Physical\_Activity, Screen\_Time, Educational\_Tech\_Use, Peer\_Group, Bullying, Study\_Space, Lack\_of\_Interest, Time\_Wasted\_on\_Social\_Media, Sports\_Participation

*   **Target Variable:** Grades (A, B, C)
                                                                     
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
