
# Titanic Survival Prediction

## Project Overview
This project uses machine learning to predict whether a passenger survived the Titanic disaster based on passenger information.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Dataset
The project uses local Titanic CSV files:
- `train.csv` — training data, including survival labels
- `test.csv` — test data without survival labels
- `gender_submission.csv` — sample submission format

## Machine Learning Workflow
1. Load and explore the dataset.
2. Handle missing values.
3. Perform feature engineering.
4. Preprocess numerical and categorical features.
5. Compare multiple machine learning models.
6. Tune the Random Forest model.
7. Evaluate the model on a validation split.
8. Generate predictions for the test dataset.

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the project:

```bash
python titanic_model.py
```

The script generates evaluation results, charts, and a CSV file containing test predictions.

## Note
The Kaggle test dataset does not include actual survival labels, so its true prediction accuracy cannot be calculated locally without those labels.
