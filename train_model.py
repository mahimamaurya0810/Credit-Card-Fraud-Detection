import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import joblib

# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Select features used in the project
features = [
    "V1", "V2", "V3", "V4", "V5",
    "V7", "V9", "V10", "V11", "V12",
    "V14", "V16", "V17", "V18"
]

X = df[features]
y = df["Class"]

# 4. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=25,
    stratify=y
)

# 5. Random Under Sampling
undersampler = RandomUnderSampler(random_state=25)

balanced_x_train, balanced_y_train = undersampler.fit_resample(
    x_train,
    y_train
)

# 6. Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=25
)

model.fit(balanced_x_train, balanced_y_train)

# 7. Save trained model
joblib.dump(model, "model.joblib")

# 8. Save feature names
joblib.dump(features, "features.joblib")

print("✅ Model training completed!")
print("✅ model.joblib created successfully!")
print("✅ features.joblib created successfully!")