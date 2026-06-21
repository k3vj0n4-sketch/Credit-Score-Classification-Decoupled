import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")


TARGET = "Credit_Score"

NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

CATEGORICAL_FEATURES = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"]


class CreditScorePreprocessor:

    def __init__(self):
        self.num_imputer = None
        self.cat_imputer = None
        self.le_dict = {}
        self.scaler = None
        self.label_encoder = None
        self.is_fitted = False

    @staticmethod
    def _parse_credit_history(val):
        if pd.isna(val):
            return np.nan
        match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
        if match:
            years, months = int(match.group(1)), int(match.group(2))
            return years * 12 + months
        return np.nan

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data.drop(columns=[c for c in DROP_COLS if c in data.columns],
                  inplace=True, errors="ignore")

        float_cols = ["Annual_Income", "Changed_Credit_Limit",
                      "Outstanding_Debt", "Amount_invested_monthly"]
        int_cols = ["Num_of_Loan", "Num_of_Delayed_Payment"]

        for col in float_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        for col in int_cols:
            if col in data.columns:
                data[col] = (
                    pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
                )

        if "Age" in data.columns:
            data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
            data["Age"] = data["Age"].clip(lower=18, upper=100)

        if "Occupation" in data.columns:
            data["Occupation"] = data["Occupation"].replace("_______", np.nan)
        if "Credit_Mix" in data.columns:
            data["Credit_Mix"] = data["Credit_Mix"].replace("_", np.nan)
        if "Payment_Behaviour" in data.columns:
            data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        if 'Payment_of_Min_Amount' in data.columns:
            data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].replace('NM', 'No')

        if "Credit_History_Age" in data.columns:
            data["Credit_History_Age"] = (
                data["Credit_History_Age"]
                .apply(self._parse_credit_history)
                .fillna(0)
                .astype(int)
            )

        clip_cols = [
            "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
            "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
        ]
        for col in clip_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = max(0, Q1 - 3 * IQR)
                upper = Q3 + 3 * IQR
                data[col] = data[col].clip(lower=lower, upper=upper)

        return data

    def fit_transform(self, data_clean: pd.DataFrame):
        X = data_clean[ALL_FEATURES].copy()
        y = data_clean[TARGET].copy()

        self.num_imputer = SimpleImputer(strategy="median")
        X[NUMERIC_FEATURES] = self.num_imputer.fit_transform(X[NUMERIC_FEATURES])

        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        X[CATEGORICAL_FEATURES] = self.cat_imputer.fit_transform(X[CATEGORICAL_FEATURES])

        self.le_dict = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.le_dict[col] = le

        self.scaler = StandardScaler()
        X[NUMERIC_FEATURES] = self.scaler.fit_transform(X[NUMERIC_FEATURES])

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        self.is_fitted = True
        return X, y_enc

    def transform(self, data_clean: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor belum di-fit. Jalankan fit_transform() dahulu.")

        X = data_clean[ALL_FEATURES].copy()
        has_target = TARGET in data_clean.columns

        X[NUMERIC_FEATURES] = self.num_imputer.transform(X[NUMERIC_FEATURES])
        X[CATEGORICAL_FEATURES] = self.cat_imputer.transform(X[CATEGORICAL_FEATURES])

        for col in CATEGORICAL_FEATURES:
            le = self.le_dict[col]
            X[col] = X[col].astype(str).map(lambda v, le=le: v if v in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

        X[NUMERIC_FEATURES] = self.scaler.transform(X[NUMERIC_FEATURES])

        y_enc = self.label_encoder.transform(data_clean[TARGET]) if has_target else None
        return X, y_enc


class ModelTrainer:
    

    MODEL_REGISTRY = {
        "Logistic Regression": (
            LogisticRegression,
            {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42},
        ),
        "Decision Tree": (
            DecisionTreeClassifier,
            {"max_depth": 15, "min_samples_split": 10,
             "min_samples_leaf": 5, "random_state": 42},
        ),
        "Random Forest": (
            RandomForestClassifier,
            {"n_estimators": 150, "max_depth": 20, "min_samples_split": 5,
             "min_samples_leaf": 2, "class_weight": "balanced",
             "random_state": 42, "n_jobs": -1},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier,
            {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "min_samples_leaf": 5, "random_state": 42},
        ),
    }

    def __init__(self, model_name: str):
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' tidak terdaftar di MODEL_REGISTRY.")
        self.model_name = model_name
        model_cls, params = self.MODEL_REGISTRY[model_name]
        self.params = params
        self.model = model_cls(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model


class ModelEvaluator:
    

    def __init__(self, model, model_name: str, params: dict = None):
        self.model = model
        self.model_name = model_name
        self.params = params or {}

    def evaluate(self, X_test, y_test, classes, log_to_mlflow: bool = True):
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=classes)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n=== {self.model_name} ===")
        print(f"Accuracy   : {acc:.4f}")
        print(f"F1 Macro   : {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(report)

        if log_to_mlflow:
            if self.params:
                mlflow.log_params(self.params)
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.sklearn.log_model(self.model, name="model")

        return {
            "model_name": self.model_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "y_pred": y_pred,
            "confusion_matrix": cm,
        }


class CreditScorePipeline:
    

    MODEL_NAMES = ["Logistic Regression", "Decision Tree",
                   "Random Forest", "Gradient Boosting"]

    def __init__(self, csv_path: str,
                mlflow_tracking_uri: str = "sqlite:///mlflow.db",
                experiment_name: str = "credit_score_classification"):
        self.csv_path = csv_path
        self.preprocessor = CreditScorePreprocessor()
        self.results = {}
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        print(f"Data loaded: {df.shape}")
        return df

    def run(self, test_size: float = 0.2, random_state: int = 42):
        df = self.load_data()

        data_clean = self.preprocessor.clean_data(df)
        print("Shape setelah cleaning:", data_clean.shape)

        train_df, test_df = train_test_split(
            data_clean, test_size=test_size, random_state=random_state,
            stratify=data_clean[TARGET],
        )
        print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

        X_train, y_train = self.preprocessor.fit_transform(train_df)
        X_test, y_test = self.preprocessor.transform(test_df)
        classes = self.preprocessor.label_encoder.classes_
        print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
        print("Classes:", list(classes))

        for model_name in self.MODEL_NAMES:
            with mlflow.start_run(run_name=model_name):
                trainer = ModelTrainer(model_name)
                model = trainer.train(X_train, y_train)

                evaluator = ModelEvaluator(model, model_name, trainer.params)
                metrics = evaluator.evaluate(X_test, y_test, classes)

                self.results[model_name] = metrics
                self.trained_models[model_name] = model

        results_df = pd.DataFrame({
            name: {"Accuracy": m["accuracy"], "F1 Macro": m["f1_macro"]}
            for name, m in self.results.items()
        }).T.sort_values("F1 Macro", ascending=False)

        print("\n=== Perbandingan Semua Model ===")
        print(results_df.to_string())

        self.best_model_name = results_df.index[0]
        self.best_model = self.trained_models[self.best_model_name]
        print(f"\nModel Terbaik: {self.best_model_name} "
              f"(F1 Macro: {results_df.iloc[0]['F1 Macro']:.4f})")

        return self.best_model_name, self.best_model

    def save_pickle(self, output_path: str = "model_pipeline.pkl"):
        if self.best_model is None:
            raise RuntimeError("Belum ada model. Jalankan run() dahulu.")

        artifact = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "num_imputer": self.preprocessor.num_imputer,
            "cat_imputer": self.preprocessor.cat_imputer,
            "le_dict": self.preprocessor.le_dict,
            "scaler": self.preprocessor.scaler,
            "label_encoder": self.preprocessor.label_encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "all_features": ALL_FEATURES,
            "classes": list(self.preprocessor.label_encoder.classes_),
        }
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)

        print(f"\nPickle tersimpan di: {Path(output_path).resolve()}")


if __name__ == "__main__":
    DATA_PATH = "data_C.csv"         
    OUTPUT_PKL = "model_pipeline.pkl"

    pipeline = CreditScorePipeline(csv_path=DATA_PATH)
    pipeline.run()
    pipeline.save_pickle(OUTPUT_PKL)
