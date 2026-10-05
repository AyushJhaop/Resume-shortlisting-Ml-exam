# Resume Shortlisting Prediction — Machine Learning Case Study

## 1. Problem
Predict whether a candidate should be shortlisted for interview using:
- Years of experience
- Education level
- Relevant skill count
- Certification count
- Previous industry

The model is a decision-support prototype, not a replacement for recruiters.

## 2. Models
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

## 3. Workflow
1. Generate synthetic recruitment data.
2. Perform EDA.
3. Handle missing values.
4. Encode categorical variables.
5. Scale numerical variables.
6. Split data into train/test sets.
7. Compare five classification algorithms.
8. Use 5-fold stratified cross-validation.
9. Evaluate Accuracy, Precision, Recall, F1 and Confusion Matrix.
10. Evaluate group-wise performance by education and previous industry.
11. Save the best model.
12. Deploy it using Streamlit.

## 4. Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py
python train.py
streamlit run app.py
```

On Windows activation is usually:

```bash
.venv\Scripts\activate
```

## 5. Synthetic-data explanation for viva
The dataset is synthetic because no standard public dataset exactly matches the required fields.
The target was generated probabilistically from realistic relationships: experience,
relevant skills, certifications and education increase the probability of shortlisting,
while industry contributes a smaller contextual effect. Random noise prevents the target
from being a perfect deterministic rule. Missing values were intentionally introduced in
experience and certification fields to demonstrate preprocessing.

## 6. Important ethical point
Gender, age, religion and other sensitive attributes are not included. However, this does
not automatically prove that the model is fair. The case study therefore reports performance
across non-sensitive groups available in the dataset (education and industry) and clearly
states that synthetic-data evaluation cannot establish real-world fairness.

## 7. Selection rule
The code selects the model with the highest mean 5-fold cross-validated F1 score.
Mean CV precision is used as a tie-breaker. Test-set precision is reported separately.
This avoids choosing a model from one test result alone.
