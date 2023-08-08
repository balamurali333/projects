import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import urllib.request

# Step 1: Load and preprocess the data
data_url = "https://raw.githubusercontent.com/simranjeet97/Top-Machine-Learning-Algorithms-Python/main/insurance_data.csv"
data = pd.read_csv(data_url)
data = data.drop(columns=['_c39', 'policy_number', 'policy_bind_date', 'incident_date', 'incident_location'])
data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
X = data.drop(columns=['fraud_reported'])
y = data['fraud_reported']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocess categorical features using one-hot encoding
categorical_features = ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level', 'insured_occupation',
                        'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type', 'incident_severity',
                        'authorities_contacted', 'incident_state', 'incident_city', 'property_damage',
                        'police_report_available', 'auto_make', 'auto_model']

auto_year_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
auto_years = data['auto_year'].unique().reshape(-1, 1)
auto_year_encoder.fit(auto_years)
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Step 4: Apply feature scaling using MaxAbsScaler for sparse data
scaler = MaxAbsScaler() # scales each feature to have an absolute maximum of 1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Hyperparameter tuning using GridSearchCV with XGBoost
param_grid_xgb = {
    'learning_rate': [0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Step 6: Evaluate the XGBoost model's performance
y_pred_xgb = best_xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# Step 7: Hyperparameter tuning using GridSearchCV with RandomForestClassifier
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']  # Use 'sqrt' instead of 'auto'
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_

# Step 8: Evaluate the RandomForest model's performance
y_pred_rf = best_rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Step 9: Hyperparameter tuning using GridSearchCV with Support Vector Classifier (SVC)
param_grid_svc = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

svc_model = SVC(random_state=42)
grid_search_svc = GridSearchCV(svc_model, param_grid_svc, cv=3, n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_svc_model = grid_search_svc.best_estimator_

# Step 10: Evaluate the SVC model's performance
y_pred_svc = best_svc_model.predict(X_test)

accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision_svc = precision_score(y_test, y_pred_svc)
recall_svc = recall_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc)

# Create the Ensemble model using majority voting
voting_clf = VotingClassifier(
    estimators=[('xgb', best_xgb_model), ('rf', best_rf_model), ('svc', best_svc_model)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)

# Step 12: Evaluate the Ensemble model's performance
y_pred_ensemble = voting_clf.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

print("\nEnsemble Model:")
print("Accuracy:", accuracy_ensemble)
print("Precision:", precision_ensemble)
print("Recall:", recall_ensemble)
print("F1-Score:", f1_ensemble)

# Tkinter GUI for user input and fraud prediction
root = tk.Tk()
root.title("Fraud Detection")

# Create a scrollable frame
canvas = tk.Canvas(root)
scroll_frame = tk.Frame(canvas)

scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

fields = [
    'months_as_customer', 'age', 'policy_state', 'policy_csl',
    'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip',
    'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship',
    'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_state', 'incident_city','incident_hour_of_the_day',
    'number_of_vehicles_involved', 'property_damage','bodily_injuries', 'witnesses', 'police_report_available',
    'total_claim_amount', 'injury_claim','property_claim', 'vehicle_claim',
    'auto_make', 'auto_model', 'auto_year'
]

entry_widgets = {}

# Create input fields in the scrollable frame
for i, field in enumerate(fields): # creates input fields for various features
    row = tk.Frame(scroll_frame)
    label = tk.Label(row, width=20, text=field, anchor='w')
    entry = tk.Entry(row)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entry_widgets[field] = entry

def preprocess_input(user_input): # This returns the preprocessed input data.
    sample_df = pd.DataFrame(user_input, index=[0])
    sample_df = sample_df.reindex(columns=fields, fill_value=0)

    auto_year = int(sample_df['auto_year'].values[0])
    auto_year_encoded = auto_year_encoder.transform([[auto_year]])

    sample_df_encoded = preprocessor.transform(sample_df)
    sample_df_encoded = scaler.transform(sample_df_encoded)

    # Concatenate the auto_year_encoded array to the end of the sample_df_encoded array
    sample_df_encoded = np.hstack((sample_df_encoded, auto_year_encoded))
    return sample_df_encoded

def predict_fraud():
    user_input = {field: entry.get() for field, entry in entry_widgets.items()}
    try:
        sample_df = pd.DataFrame(user_input, index=[0])
        sample_df = sample_df.reindex(columns=fields, fill_value=0)  # Use the original order of fields
        sample_df_encoded = preprocessor.transform(sample_df)
        sample_df_scaled = scaler.transform(sample_df_encoded)

        prediction = voting_clf.predict(sample_df_scaled)[0]
        result = {0: 'Given Data is "Not Fraud"', 1: 'Given Data is "Fraud"'}[prediction]

        # Since you're using hard voting, you won't have predict_proba available
        f1 = f1_ensemble  # Use the ensemble F1-score
        precision = precision_ensemble
        recall = recall_ensemble

        messagebox.showinfo("Prediction Result", f"The input data is predicted as: {result}\n"
                                                 f"Accuracy: {accuracy_ensemble:.2f}\n"
                                                 f"F1-Score: {f1:.2f}\n"
                                                 f"Precision: {precision:.2f}\n"
                                                 f"Recall: {recall:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

predict_button = tk.Button(root, text="Predict Fraud", command=predict_fraud)
predict_button.pack()

root.mainloop()