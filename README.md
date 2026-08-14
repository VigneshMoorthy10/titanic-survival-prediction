# Titanic Survival Prediction

Predicting which passengers survived the Titanic disaster using machine learning, built on the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Problem
Given passenger data (age, sex, class, fare, family size, etc.), predict whether each passenger survived (1) or did not survive (0).

## Approach

**1. Exploratory Data Analysis**
Visualized survival patterns across sex, passenger class, age, and fare. Found that:
- Women had a much higher survival rate than men
- 1st class passengers survived at higher rates than 2nd or 3rd class
- Young children had noticeably better survival odds than other age groups
- Survivors tended to have paid higher fares on average
- Overall, survival correlated strongly with social status and gender rather than being random
- - Survival rate by family size revealed a clear pattern: passengers traveling alone had only 30% survival, small families (3-4 people) peaked at 55-72%, while large families (7+) had 0-33% survival — small groups could move and help each other efficiently, while large families may have struggled to stay together or fit in lifeboats
- Even within 1st class, survival rate differed drastically by sex — 96.8% for women vs 36.9% for men — showing gender was a stronger predictor than wealth/class alone
- - Even among the top 5 highest-paying passengers (all 1st class), one still did not survive — wealth improved odds significantly but wasn't an absolute guarantee
- Young boys (Title = "Master") survived at 57.5%, roughly 3x the survival rate of adult men (~19%) — showing "women and children first" extended specifically to boys, not just women

- **Data Validation**
Before modeling, combined `train` and `test` using `pd.concat()` to check for distribution shift — confirmed the average passenger age was nearly identical between the two sets (29.39 vs 29.68), indicating a fair, representative split. This kind of sanity check helps catch cases where a model might perform well in training but poorly on unseen data due to mismatched distributions.

**2. Data Cleaning**
- Filled missing `Age` values using the median age per passenger title (Mr/Mrs/Miss/etc.) rather than a single flat average, for a more accurate estimate
- Dropped `Cabin` (over 75% missing — too sparse to reliably impute)
- Filled the 2 missing `Embarked` values with the most common port

**3. Feature Engineering**
- Extracted `Title` (Mr, Mrs, Miss, Master, Rare) from passenger names
- Created `FamilySize` from `SibSp` + `Parch` + 1
- Encoded `Sex` and `Embarked` as numeric values

**4. Modeling**
Trained and compared three classifiers on a held-out validation split:

| Model | Validation Accuracy |
|---|---|
| Logistic Regression | 78.2% |
| Decision Tree | 79.9% |
| **Random Forest** | **84.4%** |

Random Forest performed best and was used for the final submission.

**5. Result**
Official Kaggle leaderboard score: **0.74162**

Note: this is lower than the 84.4% validation accuracy, which is expected — the validation split came from the same training data distribution the model was fit on, while the leaderboard evaluates on entirely unseen passengers. The gap also suggests some overfitting from the default Random Forest settings.

## What I'd try next
- Hyperparameter tuning (max_depth, min_samples_split) to reduce overfitting
- K-fold cross-validation instead of a single train/validation split, for a more reliable accuracy estimate
- Try gradient boosting models (XGBoost/LightGBM) for comparison
- Ensemble multiple models together

## Files
- `titanic-survival-prediction.ipynb` — full notebook (EDA, cleaning, feature engineering, modeling)
- `submission.csv` — final predictions submitted to Kaggle

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn — run in a Kaggle Notebook
