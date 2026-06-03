# 🎓 Student Grade Prediction ML Pipeline

A professional machine learning project for predicting student grades using multiple classification algorithms.  
This project reads student performance data from a CSV file, trains several machine learning models, compares their performance, selects the best model, and saves reports and the final trained model automatically.

---

## 🚀 Project Overview

This project predicts a student's final **grade** based on performance-related features such as:

- Weekly self-study hours
- Attendance percentage
- Class participation
- Total score

The system trains and evaluates multiple machine learning models and selects the best-performing model based on evaluation metrics.

---

## ✨ Features

- ✅ Clean and professional ML pipeline
- ✅ Automatic dataset loading
- ✅ Duplicate row removal
- ✅ Missing value handling
- ✅ Automatic feature and target separation
- ✅ Label encoding for grade values
- ✅ Train/test splitting with stratification
- ✅ Prevents data leakage using `Pipeline`
- ✅ Uses preprocessing safely inside model pipelines
- ✅ Compares multiple ML algorithms
- ✅ Calculates accuracy, precision, recall, weighted F1, and macro F1
- ✅ Saves best trained model automatically
- ✅ Saves model comparison report
- ✅ Saves classification report
- ✅ Saves confusion matrix
- ✅ Saves training metadata
- ✅ Supports future predictions using saved model
- ✅ Suitable for large datasets

---

## 🧠 Machine Learning Models Used

The project compares the following models:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification model suitable for multi-class classification |
| Decision Tree | Tree-based decision-making model |
| Random Forest | Ensemble model using multiple decision trees |
| Hist Gradient Boosting | High-performance gradient boosting model |
| Gaussian Naive Bayes | Probabilistic classification model |
| Bernoulli Naive Bayes | Naive Bayes model for binary-style features |
| KNN | Optional model, disabled by default for large datasets |

---

## 📂 Project Structure

```txt
Studentp/
│
├── student_performance.csv
├── student.py
├── requirements.txt
│
├── reports/
│   ├── model_comparison.csv
│   ├── best_classification_report.txt
│   ├── best_confusion_matrix.csv
│   └── training_metadata.json
│
└── models/
    └── best_student_grade_model.joblib
````

---

## 📊 Dataset Format

The project expects a CSV file named:

```txt
student_performance.csv
```

Example dataset columns:

```txt
student_id
weekly_self_study_hours
attendance_percentage
class_participation
total_score
grade
```

Example:

```csv
student_id,weekly_self_study_hours,attendance_percentage,class_participation,total_score,grade
1,12,88,7,85,A
2,5,65,4,62,C
3,2,45,2,39,F
4,9,78,6,74,B
```

---

## 🎯 Target Column

The target column is:

```txt
grade
```

Example grade values:

```txt
A
B
C
D
F
```

The program automatically converts these grade labels into numeric values using `LabelEncoder`.

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
https://github.com/TutorialsAndroid/Studentp.git
cd Studentp
```

Or simply place these files in one folder:

```txt
student_performance.csv
student.py
requirements.txt
```

---

### 2. Create `requirements.txt`

```txt
numpy
pandas
scikit-learn
joblib
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the training file:

```bash
python student.py
```

After successful execution, the program will:

1. Load the dataset
2. Clean the dataset
3. Split data into training and testing sets
4. Train multiple ML models
5. Compare all models
6. Select the best model
7. Save reports
8. Save the best trained model

---

## 📈 Output Example

After running the program, you will see output similar to this:

```txt
================ MODEL COMPARISON ================

                 model  train_rows  test_rows  accuracy  precision_weighted  recall_weighted  f1_weighted  f1_macro
         Random Forest     800000    200000     0.9987              0.9987           0.9987       0.9987    0.9979
Hist Gradient Boosting     800000    200000     0.9975              0.9976           0.9975       0.9975    0.9961
   Logistic Regression     800000    200000     0.9852              0.9854           0.9852       0.9851    0.9748

================ BEST MODEL ================

Best Model: Random Forest
Accuracy: 0.9987
Weighted F1: 0.9987
Macro F1: 0.9979
```

---

## 📁 Generated Reports

After training, the following files are automatically generated.

### `reports/model_comparison.csv`

Contains performance comparison of all trained models.

```csv
model,train_rows,test_rows,accuracy,precision_weighted,recall_weighted,f1_weighted,f1_macro
Random Forest,800000,200000,0.9987,0.9987,0.9987,0.9987,0.9979
```

---

### `reports/best_classification_report.txt`

Contains detailed class-wise performance.

Example:

```txt
              precision    recall  f1-score   support

           A       1.00      1.00      1.00    109729
           B       1.00      1.00      1.00     51635
           C       1.00      1.00      1.00     28396
           D       1.00      1.00      1.00      8999
           F       1.00      1.00      1.00      1241
```

---

### `reports/best_confusion_matrix.csv`

Contains the confusion matrix of the best model.

Example:

```csv
, predicted_A, predicted_B, predicted_C, predicted_D, predicted_F
actual_A,109729,0,0,0,0
actual_B,0,51635,0,0,0
actual_C,0,0,28396,0,0
actual_D,0,0,0,8999,0
actual_F,0,0,0,0,1241
```

---

### `reports/training_metadata.json`

Contains useful training information.

Example:

```json
{
    "best_model": "Random Forest",
    "feature_columns": [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation",
        "total_score"
    ],
    "target_classes": [
        "A",
        "B",
        "C",
        "D",
        "F"
    ],
    "train_rows": 800000,
    "test_rows": 200000,
    "include_total_score": true
}
```

---

## 💾 Saved Model

The best trained model is saved inside:

```txt
models/best_student_grade_model.joblib
```

This file contains:

* Best trained ML model
* Label encoder
* Feature column names
* Target column name
* Best model name
* Configuration metadata

---

## 🔮 Predict Using Saved Model

You can use the saved model to predict a new student's grade.

Example:

```python
from train_student_grade_model import predict_with_saved_model

prediction = predict_with_saved_model({
    "weekly_self_study_hours": 10,
    "attendance_percentage": 85,
    "class_participation": 7,
    "total_score": 82
})

print("Predicted Grade:", prediction)
```

Output:

```txt
Predicted Grade: A
```

---

## ⚠️ Important Dataset Note

Your dataset contains both:

```txt
total_score
grade
```

Usually, `grade` is directly calculated from `total_score`.

Example:

```txt
total_score >= 90  → A
total_score >= 80  → B
total_score >= 70  → C
```

Because of this, the model may achieve very high accuracy.

For a more realistic prediction system, you can disable `total_score`.

Open `train_student_grade_model.py` and change:

```python
INCLUDE_TOTAL_SCORE: bool = True
```

to:

```python
INCLUDE_TOTAL_SCORE: bool = False
```

Then the model will predict grade using only:

```txt
weekly_self_study_hours
attendance_percentage
class_participation
```

This gives a more realistic ML prediction because the model will not directly depend on final marks.

---

## 🛡️ Why This Version Is Better

The old version worked, but it had some issues:

| Old Version                          | New Version                           |
| ------------------------------------ | ------------------------------------- |
| Scaled full dataset before splitting | Uses pipeline to avoid data leakage   |
| Used `student_id` as a feature       | Drops `student_id` automatically      |
| Only printed accuracy                | Generates multiple evaluation metrics |
| No saved model                       | Saves best model automatically        |
| No reports                           | Saves CSV, TXT, and JSON reports      |
| Repeated variable names              | Cleaner structured code               |
| No missing value handling            | Handles missing values safely         |
| Manual model comparison              | Automatic model ranking               |
| Less reusable                        | Supports future predictions           |

---

## 📌 Data Leakage Prevention

The upgraded code uses `Pipeline` and `ColumnTransformer`.

This means preprocessing steps like scaling and missing value handling are learned only from the training data.

This is important because if preprocessing is applied before train/test split, information from test data can accidentally leak into training.

The upgraded code avoids this problem.

---

## 🧪 Evaluation Metrics

The project evaluates models using:

| Metric             | Meaning                                                    |
| ------------------ | ---------------------------------------------------------- |
| Accuracy           | Overall correct predictions                                |
| Precision Weighted | Correct positive predictions considering class size        |
| Recall Weighted    | Ability to find actual class labels considering class size |
| F1 Weighted        | Balanced score considering class imbalance                 |
| F1 Macro           | Average F1 across all classes equally                      |

---

## ⚡ Performance Note

Your dataset has around 10 lakh rows, so some models may take more time.

KNN is disabled by default because it can be slow on large datasets.

To enable KNN, change:

```python
ENABLE_KNN: bool = False
```

to:

```python
ENABLE_KNN: bool = True
```

KNN will use a sample of the dataset for better performance.

---

## 🧾 Changelog

### v2.0.0 — Professional ML Pipeline Upgrade

#### Added

* Added professional project structure
* Added configuration using `Config` dataclass
* Added automatic dataset validation
* Added automatic duplicate removal
* Added missing value handling
* Added safe preprocessing using `Pipeline`
* Added `ColumnTransformer` for numeric and categorical features
* Added automatic feature detection
* Added label encoding for target column
* Added stratified train/test split
* Added multiple model training system
* Added automatic model comparison
* Added weighted precision score
* Added weighted recall score
* Added weighted F1 score
* Added macro F1 score
* Added classification report generation
* Added confusion matrix generation
* Added best model selection
* Added best model saving using `joblib`
* Added metadata saving in JSON format
* Added future prediction helper function
* Added support for disabling `total_score`
* Added optional KNN mode for large datasets
* Added detailed logging

#### Improved

* Improved model training structure
* Improved code readability
* Improved project maintainability
* Improved error handling
* Improved large dataset support
* Improved evaluation quality
* Improved feature handling
* Improved reporting system
* Improved prediction reusability

#### Fixed

* Fixed possible data leakage issue
* Fixed use of `student_id` as prediction feature
* Fixed repeated variable naming issues
* Fixed lack of target validation
* Fixed missing report generation
* Fixed lack of saved model support
* Fixed poor scalability for large datasets
* Fixed unsafe preprocessing before train/test split

---

### v1.0.0 — Initial Basic ML Script

#### Added

* Added CSV loading using pandas
* Added grade encoding using LabelEncoder
* Added feature scaling using StandardScaler
* Added train/test split
* Added Logistic Regression model
* Added KNN model
* Added Decision Tree model
* Added Random Forest model
* Added Gaussian Naive Bayes model
* Added Bernoulli Naive Bayes model
* Added accuracy score printing

#### Limitations

* Preprocessing was applied before train/test split
* Only accuracy was calculated
* No model saving
* No report generation
* No confusion matrix
* No classification report
* No production-level structure
* No future prediction function
* No automatic best model saving

---

## 🧑‍💻 Author

Developed as a professional machine learning pipeline for student grade prediction.

---

## 📜 License

This project can be used for learning, academic projects, portfolio projects, and machine learning practice.

---

## ⭐ Recommended Use Cases

This project is useful for:

* Machine learning practice
* Student performance prediction
* Academic ML projects
* Portfolio projects
* Classification model comparison
* Learning sklearn pipelines
* Understanding model evaluation
* Demonstrating production-style ML structure

---

## ✅ Final Result

This upgraded project is no longer just a simple ML script.

It is now a cleaner and more professional machine learning pipeline that can:

```txt
Load data → Clean data → Train models → Compare results → Save reports → Save best model → Predict future grades
```