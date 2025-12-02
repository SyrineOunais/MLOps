from flask import Flask, render_template, request
import pandas as pd
from model_pipeline import preprocess_data, load_model, predict, prepare_data, save_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Load initial model
model = load_model("random_forest_loan_model.pkl")
trained_columns = model.feature_names_in_

# ----------------------------
# Home & Predict Form
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model, trained_columns

    if request.method == "POST":
        # Collect form data
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        # Convert numeric fields
        for col in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # Preprocess
        df_processed = preprocess_data(df, is_train=False)

        if "Loan_ID" in df_processed:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Add missing columns
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        df_processed = df_processed[trained_columns]

        # Predict
        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return render_template("result.html", result=result)

    return render_template("index.html")

# ----------------------------
# Retrain Form
# ----------------------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global model, trained_columns

    if request.method == "POST":
        # Get hyperparameters
        n_estimators = int(request.form.get("n_estimators", 100))
        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None
        min_samples_split = int(request.form.get("min_samples_split", 2))

        # Load data
        data = prepare_data("data/train.csv", "data/test.csv")
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        # Evaluate old model
        old_pred = model.predict(X_test)
        old_acc = accuracy_score(y_test, old_pred)
        old_cm = confusion_matrix(y_test, old_pred).tolist()

        # Train new RandomForest model
        new_model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        new_model.fit(X_train, y_train)

        # Evaluate new model
        new_pred = new_model.predict(X_test)
        new_acc = accuracy_score(y_test, new_pred)
        new_cm = confusion_matrix(y_test, new_pred).tolist()

        # Compare & keep best
        if new_acc > old_acc:
            save_model(new_model, "random_forest_loan_model.pkl")
            model = load_model("random_forest_loan_model.pkl")
            trained_columns = model.feature_names_in_
            chosen = "new"
        else:
            save_model(new_model, "random_forest_rejected.pkl")
            chosen = "old"

        comparison = {
            "old_model": {"accuracy": old_acc, "confusion_matrix": old_cm},
            "new_model": {"accuracy": new_acc, "confusion_matrix": new_cm, "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}},
            "better_model": chosen
        }

        return render_template("retrain.html", comparison=comparison)

    return render_template("retrain.html", comparison=None)

if __name__ == "__main__":
    app.run(debug=True)
