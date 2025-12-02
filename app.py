from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd

from model_pipeline import (
    preprocess_data,
    load_model,
    predict,
    prepare_data,      
    train_model,       
    evaluate_model,   
    save_model,       
    retrain_model
)



app = FastAPI()

# ------------------------------
# Schéma pour l'endpoint /retrain
# ------------------------------
class RetrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42

model = load_model("random_forest_loan_model.pkl")
trained_columns = model.feature_names_in_

class LoanData(BaseModel):
    Loan_ID: str
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.post("/predict")
def predict_loan(data: LoanData):
    try:
        df = pd.DataFrame([data.dict()])

        # Apply training preprocessing
        df_processed = preprocess_data(df, is_train=False)

        # Drop Loan_ID
        if "Loan_ID" in df_processed:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Add missing columns with 0
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Reorder columns exactly as random_forest_loan_model expects
        df_processed = df_processed[trained_columns]

        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return {"Loan_ID": data.Loan_ID, "prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------------------
# Endpoint /retrain
# ------------------------------
@app.post("/retrain")
def retrain_model_endpoint(params: RetrainParams, background_tasks: BackgroundTasks):
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres en tâche de fond
    """
    def retrain_task(params: RetrainParams):
        global model, trained_columns
        try:
            data = prepare_data("data/train.csv", "data/test.csv")
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            # Entraîner le modèle
            new_model = retrain_model(
                X_train, y_train,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                random_state=params.random_state
            )

            # Évaluer
            results = evaluate_model(new_model, X_test, y_test)
            print(f"Modèle réentraîné avec accuracy: {results['accuracy']:.4f}")

            # Sauvegarder
            save_model(new_model,"random_forest_loan_model_retrained.pkl")

            # Mettre à jour le modèle en mémoire
            model = new_model
            trained_columns = model.feature_names_in_

        except Exception as e:
            print(f"Erreur lors du réentraînement: {e}")

    background_tasks.add_task(retrain_task, params)

    return {"status": "retraining started", "hyperparameters": params.dict()}