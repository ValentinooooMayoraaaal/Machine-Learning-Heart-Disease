# Heart Disease Risk Prediction

This project was built as part of the Introduction to Machine Learning course at EFREI Paris. It explores a classic binary classification problem: predicting whether a patient is at risk (1) or not at risk (0) of heart disease, based on a set of medical attributes.

## 1. Overview

We approached this as a **binary classification problem**: given a set of medical measurements for a patient, our goal was to train a model capable of predicting the presence (`1`) or absence (`0`) of heart disease.

We worked with the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (`heart.csv`, 1,025 rows, 14 columns), which includes attributes such as:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Biological sex (1 = male, 0 = female) |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia status |
| `target` | 1 = at risk, 0 = not at risk |

## 2. Our Approach

### 2.1 Exploratory Data Analysis
We started by exploring the dataset's structure, checking for missing values and data types, and visualizing key relationships — for example, the age distribution across patients and how maximum heart rate (`thalach`) differs between at-risk and not-at-risk patients.

### 2.2 Data Preprocessing
Before training any model, we cleaned and prepared the data:
- **Missing values**: we imputed missing entries in `ca` and `thal` using the most frequent value of each column.
- **Categorical encoding**: we one-hot encoded categorical features (`cp`, `restecg`, `slope`, `thal`, `sex`, `fbs`, `exang`) so the models wouldn't misinterpret category numbers as ordered quantities.
- **Feature scaling**: we standardized numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) using `StandardScaler`, so no single feature would dominate due to its scale.
- **Train/test split**: we held out 20% of the data as a test set (stratified on the target) to evaluate our models on data they had never seen.

### 2.3 Models We Trained
We trained and compared three classification algorithms:

- **Logistic Regression** — a linear model that estimates the probability of heart disease as a weighted combination of the input features, passed through a sigmoid function.
- **Decision Tree** — a model that learns a sequence of yes/no questions on the features (e.g. "cholesterol > 240?") to reach a prediction.
- **Random Forest** — an ensemble of many decision trees, each trained on a different random subset of the data, whose predictions are combined by majority vote for a more robust result.

For each model, we used **GridSearchCV** with 5-fold cross-validation to automatically search for the best hyperparameters (e.g. tree depth, regularization strength, number of trees), optimizing for **recall** — since in a medical context, failing to flag an at-risk patient is more costly than a false alarm.

### 2.4 Evaluation Metrics
We evaluated every model on the held-out test set using:
- **Accuracy** — overall proportion of correct predictions
- **Precision** — of patients predicted at risk, how many actually were
- **Recall** — of patients actually at risk, how many we correctly identified
- **F1-score** — harmonic mean of precision and recall
- **AUC-ROC** — the model's ability to separate the two classes across all thresholds

## 3. Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree | 0.990 | 0.981 | 1.000 | 0.991 | 1.000 |
| Logistic Regression | 0.873 | 0.856 | 0.905 | 0.880 | 0.945 |

Based on F1-score, **Random Forest** came out as our best-performing model.

## 4. Interpreting Our Results

We want to be transparent about a limitation we identified in this project: a near-perfect score (100% across every metric for Random Forest) is unusually high for a real-world medical dataset, and it's a signal worth investigating rather than celebrating at face value.

When we looked closer, we found that our dataset contains **723 duplicate rows out of 1,025** — a known characteristic of this particular version of the Kaggle heart disease dataset. Because we split the data into train/test *after* these duplicates were already present, some rows in our test set are near-identical (or identical) to rows the model had already seen during training. This causes **data leakage**: the model isn't really generalizing, it's partly "recognizing" examples it was trained on.

We see this as an important takeaway rather than a flaw to hide — it's exactly the kind of check we now know to run before trusting a model's performance:
- Deduplicate the dataset *before* splitting into train/test.
- Re-evaluate all three models on the deduplicated data to get an honest estimate of generalization performance.
- Expect Random Forest and Decision Tree scores to drop closer to (and possibly below) Logistic Regression's, which — being a simpler, less flexible model — is likely less prone to overfitting on duplicated patterns.

## 5. Tools & Libraries

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — data visualization
- `scikit-learn` — preprocessing, modeling, hyperparameter tuning, and evaluation

## 6. What's Next

- Remove duplicate rows before splitting the data to get a trustworthy performance estimate.
- Add feature importance analysis (especially for Random Forest) to understand which medical attributes matter most.
- Test additional models (e.g. Gradient Boosting, SVM) for comparison.

## 7. Team

This project was built collaboratively as part of our introduction to machine learning coursework at EFREI Paris.
