# Machine-Learning-Course

Machine Learning Full Course: Employee Churn Prediction Project
Welcome to the full practical Machine Learning course where you will learn how to predict employee churn, also known as employee attrition. This is a hands-on, project-based course designed for beginner to intermediate learners who want to build real-world machine learning skills through practice.

You will work on a realistic HR analytics dataset and implement multiple machine learning models to understand which employees are at risk of leaving the company. The goal is to help organizations make data-driven decisions to improve employee retention.

What You Will Learn
This course walks you through building an end-to-end machine learning pipeline from scratch using Python.

Real World Dataset
Understand and explore a real HR dataset that includes features like:
Satisfaction level
Monthly hours worked
Department
Salary
Promotion history
Work accidents
Tenure in the company
End to End ML Workflow
Data loading and exploratory analysis
Data preprocessing (handle missing values, encode categorical variables)
Train test split
Build and train multiple machine learning models
Evaluate performance using accuracy, precision, recall, F1 score
Visualize results such as feature importance and ROC AUC
Save metrics and plots automatically
Multiple Machine Learning Models Covered
The following models are implemented step by step:

Decision Tree
Random Forest
Logistic Regression
Support Vector Machine (SVM)
k Nearest Neighbors (KNN)
XGBoost
LightGBM
Naive Bayes
Multi Layer Perceptron (MLP)
AdaBoost
Voting Classifier
Stacking Classifier
Each model is written in a modular way so you can easily compare their performance side by side.

Why This Course?
Modular code structure: clean separation between data loading, preprocessing, modeling, and evaluation.
Ready to run scripts: just execute python main.py and see the results.
Automatic output saving: all evaluation reports and visualizations are saved into a results folder.
Model comparison ready: evaluate and compare different algorithms easily.
Resume worthy project: this is a complete machine learning project that you can add to your portfolio.
Technologies Used
This course uses popular Python libraries including:

Python 3.x
Scikit-learn
XGBoost
LightGBM
Yellowbrick
Matplotlib and Seaborn
Pandas and NumPy
Folder Structure
The project follows a clean and easy-to-understand structure:



employee_churn/
│
├── data/
│   └── employee_data.csv         # Input dataset  
│
├── results/                        # Output files and visualizations   
│   ├── metrics_*.txt               # Accuracy, classification report  
│   ├── feature_importance_*.png    # Feature importance plots  
│   └── roc_auc_*.png               # ROC AUC curves  
│
├── data_import.py                  # Load and preprocess data  
├── model.py                        # Contains all ML models  
├── evaluate.py                     # Plotting and evaluation utilities  
├── main.py                         # Main script to run the project  
How to Use This Course  
Step 1: Clone the repository  

git clone https://github.com/yourusername/ml-churn-prediction-course.git 
cd ml-churn-prediction-course
Step 2: Set up environment
Install required packages:


pip install scikit-learn xgboost lightgbm yellowbrick matplotlib pandas numpy
Step 3: Place your dataset
Make sure your dataset is located at:



data/employee_data.csv  
Step 4: Run the project  


python main.py
All evaluation metrics and visualizations will be saved in the results/ folder.

Who Is This Course For?  
Aspiring data scientists and machine learning engineers  
Students looking to apply theoretical knowledge to real world problems  
Self-taught learners who want to build a strong portfolio  
HR analysts interested in predictive analytics  
Bonus for Instructors  


If you're planning to teach this course:  

Each model is self-contained and well documented  
Easy to assign as homework or lab exercises  
Comes with ready to use visuals and reports  
Want to Improve It? 

Contributions are always welcome! Whether you want to:  

Add more models  
Introduce hyperparameter tuning  
Create a Streamlit dashboard  
Add Jupyter notebook versions  
Include SHAP values for interpretability  
Feel free to open an issue or submit a pull request!  

Like This Course?
Please give it a star on GitHub if you found it helpful and worth sharing with others.

