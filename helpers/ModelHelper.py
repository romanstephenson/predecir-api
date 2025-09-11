import os
import uuid

from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, accuracy_score, roc_auc_score,classification_report
)
from sklearn.model_selection import RandomizedSearchCV
import json
from datetime import datetime
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from helpers.PredecirDataHelper import DataHelper as dhelp
from sklearn.ensemble import RandomForestClassifier
import platform
import sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from helpers.FrequencyEncoder import FrequencyEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

class ModelHelper:
    #type(model).__name__
    """
    A helper class to help with model development and more
    """
    @staticmethod
    def _display_name(estimator):
        # if it's a pipeline, show the classifier step’s name
        return (
                    estimator.named_steps["clf"].__class__.__name__ 
                        if isinstance(estimator, Pipeline) 
                        else estimator.__class__.__name__
                )

    @staticmethod
    def evaluate(y_true, y_pred, y_probs=None):
        """
        Evaluate model performance.

        Parameters:
        - y_true : Ground truth (actual labels)
        - y_pred : Predicted class labels
        - y_probs: Predicted probabilities for the positive class (optional, required for ROC-AUC)

        Returns:
        - dict of metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1 Score": round(f1_score(y_true, y_pred), 4)
        }

        # Add ROC-AUC if probabilities are available
        if y_probs is not None:
            metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_probs), 4)

        return metrics
     
    @staticmethod
    def create_model_run_folder(base_dir="models"):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_path = os.path.join(base_dir, run_id)
        os.makedirs(run_path, exist_ok=True)
        return run_path
    
    @staticmethod
    def save_model(model, filepath, metadata=None):
        """
        Save a model using joblib and create a metadata.json file alongside.

        Parameters:
        - model: The model object to save
        - filepath: Full path to save the model (.pkl)
        - metadata: Optional dict containing metadata info (e.g., version, metrics, params)
        """
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        # Save the model
        joblib.dump(model, filepath)
        print(f"Model saved to: {filepath}")

        # Save metadata
        if metadata:
            # Auto-create version if not provided
            if "version" not in metadata:
                metadata["version"] = "v" + datetime.now().strftime("%Y%m%d_%H%M%S")

            metadata["model_filename"] = os.path.basename(filepath)
            metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            metadata_path = os.path.join(dir_path, f"{ModelHelper._display_name(model)}_metadata.json")

            # with open(metadata_path, "w") as f:
            #     json.dump(metadata, f, indent=4)
            
            # Clean non-serializable objects
            def make_json_safe(obj):
                if isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [make_json_safe(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                # fallback to string for complex objects
                return str(obj)

            safe_metadata = make_json_safe(metadata)

            with open(metadata_path, "w") as f:
                json.dump(safe_metadata, f, indent=4)

            print(f"Metadata saved to: {metadata_path}")

    @staticmethod
    def load_model(filepath):
        """
        Load a joblib model from file.

        Parameters:
        - filepath: Full path to the model (.pkl)

        Returns:
        - model object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return model

    @staticmethod
    def load_metadata(folder_path):
        """
        Load metadata.json from the given folder path.

        Parameters:
        - folder_path: Path containing the metadata.json

        Returns:
        - dict with metadata
        """
        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("metadata.json not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata

    @staticmethod
    def save_metrics_csv(metrics: dict, model_name: str, output_dir: str):
        """
        Saves evaluation metrics as a CSV file.

        Parameters:
        - metrics: dict of metrics (e.g., from ModelEvaluator)
        - model_name: "XGBoost" or "RandomForest"
        - output_dir: folder where CSV will be saved
        """
        metrics_df = pd.DataFrame([metrics])
        metrics_df.insert(0, "Model", model_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{model_name}_evaluation_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to: {csv_path}")

    @staticmethod
    def save_hyperparameters_csv(params: dict, model_name: str, output_dir: str):
        """
        Saves best hyperparameters as a CSV file.

        Parameters:
        - params: dict of hyperparameters
        - model_name: "XGBoost" or "RandomForest"
        - output_dir: folder where CSV will be saved
        """
        params_df = pd.DataFrame([params])
        params_df.insert(0, "Model", model_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{model_name}_best_hyperparameters.csv")
        params_df.to_csv(csv_path, index=False)
        print(f"Saved hyperparameters to: {csv_path}")

    @staticmethod
    def apply_threshold(model, X, threshold):
        # --- guarantee DataFrame with correct column names ---------
        if not isinstance(X, pd.DataFrame):
            if hasattr(model, "feature_names_in_"):      # works for Pipeline & LGBM
                X = pd.DataFrame(X, columns=model.feature_names_in_)
            else:
                X = pd.DataFrame(X)                      # fallback: unnamed cols
        # ---------------------------------------------------------------

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            return probs, (probs >= threshold).astype(int)
        else:
            return None, model.predict(X)
    
    @staticmethod
    def hyperparameter_search(estimator, param_grid, X, y, search_type="grid", scoring='f1', n_iter=50):
        """
        Performs hyperparameter search using GridSearchCV or RandomizedSearchCV.

        Parameters:
        - estimator: The base model to tune.
        - param_grid: Parameter grid or distributions.
        - X, y: Training data.
        - search_type: 'grid' or 'random'.
        - scoring: Metric to optimize.
        - n_iter: Number of iterations for randomized search.

        Returns:
        - Best estimator and its parameters.
        """
        print(f"Tuning {ModelHelper._display_name(estimator)} using {search_type.title()}SearchCV...")

        if search_type == "grid":
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=2,
                scoring=scoring,
                n_jobs=-1,
                verbose=2,
                error_score='raise'
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=2,
                scoring=scoring,
                n_jobs=-1,
                verbose=2,
                random_state=42,
                error_score='raise'
            )
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        print(f"Training {ModelHelper._display_name(estimator)}...")
        search.fit(X, y)

        print(f"Selecting best model for: {ModelHelper._display_name(estimator)}...")
        best_model = search.best_estimator_

        print(f"Retrieving best model params for: {ModelHelper._display_name(estimator)}...")
        best_params = search.best_params_

        return best_model, best_params

    @staticmethod
    def tune_threshold(probs, y_true, metric="f1"):
        """
        Tune classification threshold based on a selected metric.

        Parameters:
        - probs: array-like, predicted probabilities for the positive class
        - y_true: array-like, true binary labels (0 or 1)
        - metric: str, one of 'f1', 'recall', or 'precision'

        Returns:
        - best_threshold: float, threshold value that gave the best metric score
        - best_score: float, the best metric score achieved
        """
        thresholds = np.arange(0.1, 0.91, 0.05)
        scores = []

        for t in thresholds:
            preds = (probs >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true, preds)
            elif metric == "recall":
                score = recall_score(y_true, preds)
            elif metric == "precision":
                score = precision_score(y_true, preds)
            else:
                raise ValueError("Unsupported metric: choose from 'f1', 'recall', 'precision'")
            scores.append(score)

        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]

        print(f"Best Threshold for {metric}: {best_threshold:.2f} — Score: {best_score:.4f}")
        return best_threshold, best_score

    @staticmethod
    def train_validate_test(
        model,
        splits,
        param_grid=None,
        base_dir="models",
        scoring="f1",
        use_grid_search=False,
        dataset_name="dataset.csv",
        threshold=0.5,
        auto_tune_threshold=False,
        tune_metric="f1",
        search_type="grid",
        n_iter=50
    ):
        """
        Trains and evaluates a model with optional hyperparameter tuning and threshold optimization.

        Parameters:
        - model: The base model.
        - splits: Dict containing X_train, X_val, X_test, y_train, y_val, y_test.
        - param_grid: Grid or distribution of hyperparameters.
        - base_dir: Base directory for saving models.
        - dataset_name: Name of the dataset.
        - threshold: Initial classification threshold.
        - use_grid_search: Whether to tune hyperparameters.
        - search_type: 'grid' or 'random'.
        - auto_tune_threshold: If True, optimizes the threshold.
        - tune_metric: Metric to optimize ('f1', 'recall', 'precision').
        - n_iter: Number of iterations for randomized search.

        Returns:
        - best_model, run_path, test_metrics, best_params
        """

        # 1. Unpack the provided splits
        X_train = splits["X_train"]
        X_val = splits["X_val"]
        X_test = splits["X_test"]
        y_train = splits["y_train"]
        y_val = splits["y_val"]
        y_test = splits["y_test"] 

        # 2. Hyperparameter  search (optional)
        if use_grid_search and param_grid:

            best_model, best_params = ModelHelper.hyperparameter_search(
                estimator=model,
                param_grid=param_grid,
                X=X_train,
                y=y_train,
                search_type=search_type, 
                scoring=tune_metric    # same as metric for threshold tuning
            )

        else:
            print(f"Training {ModelHelper._display_name(model)} without hyperparameter search...")
            model.fit(X_train, y_train)
            print(f"Selecting best model for: {ModelHelper._display_name(model)}...")
            best_model = model
            print(f"Retrieving best model params for:  {ModelHelper._display_name(model)}...")
            best_params = model.get_params()

        # 3. Evaluation with threshold
        val_probs, _ = ModelHelper.apply_threshold(best_model, X_val, threshold)
        
        if auto_tune_threshold and val_probs is not None:
            print("Auto tuning threshold")
            threshold, best_score = ModelHelper.tune_threshold(val_probs, y_val, metric=tune_metric)
            print("Auto tuning threshold complete")
        else:
            best_score = None
            print(f"Using fixed threshold: {threshold}")

        val_probs, val_preds = ModelHelper.apply_threshold(best_model, X_val, threshold)
        test_probs, test_preds = ModelHelper.apply_threshold(best_model, X_test, threshold)

        print(f"Evaluating for model: {ModelHelper._display_name(model)}")
        val_metrics = ModelHelper.evaluate(y_val, val_preds, val_probs)
        print(f"Evaluation for model complete.")

        print(f"Testing for model: {ModelHelper._display_name(model)}")
        test_metrics = ModelHelper.evaluate(y_test, test_preds, test_probs)
        print(f"Testing for model complete.")

        print(f"Validation Report for {ModelHelper._display_name(model)}:")
        print(classification_report(y_val, val_preds))
        print(f"Final Test Report {ModelHelper._display_name(model)}:")
        print(classification_report(y_test, test_preds))

        # 4. Save model and metadata
        run_path = ModelHelper.create_model_run_folder(base_dir)
        model_path = os.path.join(run_path, f"{ModelHelper._display_name(model)}_model.pkl")



        metadata = {
            "model_type": type(best_model).__name__,
            "dataset": dataset_name,
            "threshold": threshold,
            
            # Performance metrics
            "metrics": {k: float(v) for k, v in test_metrics.items()},
            "validation_metrics": {k: float(v) for k, v in val_metrics.items()},
            
            # Hyperparameters used for training
            "hyperparameters": {k: str(v) for k, v in best_params.items()},
            
            # Model development context
            "training_info": {
                "run_id": os.path.basename(run_path),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cross_validation": use_grid_search,  # or add folds info if KFold used
                "scoring_metric": scoring,
                "auto_tune_threshold": auto_tune_threshold,
                "tune_metric": tune_metric,
                "best_score_on_val": round(best_score, 4) if best_score else None
                
            },
            
            # split counts for data
            "split_counts": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test)
            },

            # Environment metadata for reproducibility
            "environment": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "sklearn_version": sklearn.__version__,
                "libraries": {
                    "pandas": pd.__version__,
                    "seaborn": sns.__version__,
                    "matplotlib": plt.matplotlib.__version__,
                    "joblib": joblib.__version__,
                }
            }
        }


        ModelHelper.save_model(best_model, model_path, metadata)

        # 5. Save confusion matrices
        dhelp.save_confusion_matrix(
            y_val, val_preds,
            ModelHelper._display_name(best_model) + " Validation Confusion Matrix",
            os.path.join(run_path, ModelHelper._display_name(best_model) + "_val_confusion_matrix.png")
        )

        dhelp.save_confusion_matrix(
            y_test, test_preds,
            ModelHelper._display_name(best_model) + " Test Confusion Matrix",
            os.path.join(run_path, ModelHelper._display_name(best_model) + "_test_confusion_matrix.png")
        )

        # Save feature importance plot (if applicable)
        dhelp.save_feature_importance_plot(best_model, X_train.columns.tolist(), run_path)

        #save precision recall curve
        pr_curve_path = os.path.join(run_path, ModelHelper._display_name(best_model) +"_precision_recall_curve.png")
        dhelp.plot_precision_recall_curve(best_model, y_val, val_probs, save_path=pr_curve_path, model_label="VotingClassifier")


        return best_model, run_path, test_metrics, splits

    @staticmethod
    def train_ensemble(
        models: dict,  # {'XGB': xgb_model, 'LGBM': lgbm_model, 'RF': rf_model}
        splits: dict,
        base_dir="models",
        dataset_name="ensemble_dataset.csv",
        threshold=0.5,
        voting = 'soft',
        auto_tune_threshold=False,
        tune_metric="f1",
        ensemble_type = "voting",  # 'voting' or 'stacking'
        weights = None, # voting weights
        final_estimator = None
    ):
        """
        Trains a VotingClassifier ensemble from a set of base models, evaluates it,
        optionally tunes the classification threshold, and saves the trained model,
        evaluation reports, and visualizations.

        Parameters:
        - models: Dictionary of base model names and model objects.
        - splits: Dictionary with keys 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'.
        - base_dir: Directory where the model and outputs will be stored.
        - dataset_name: Name of the dataset (used in metadata).
        - threshold: Initial threshold for classification; overridden if auto_tune_threshold is True.
        - voting: Voting strategy ('soft' or 'hard') for the ensemble.
        - auto_tune_threshold: If True, selects the best threshold based on the tune_metric.
        - tune_metric: Metric to optimize when auto-tuning the threshold ('f1', 'recall', 'precision').

        Returns:
        - ensemble_model: The trained VotingClassifier model.
        - run_path: Path to the directory where outputs were saved.
        - test_metrics: Dictionary of evaluation metrics on the test set.
        """

        # Unpack splits
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_val = splits['X_val']
        y_val = splits['y_val']
        X_test = splits['X_test']
        y_test = splits['y_test']

        # Create ensemble
        estimators = [(name, model) for name, model in models.items()]

        # === Build Ensemble === Allow choice of voting or stacking
        if ensemble_type == "voting":
            ensemble_model = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
        elif ensemble_type == "stacking":
            ensemble_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator or LogisticRegression(max_iter=1000))
        else:
            raise ValueError("ensemble_type must be 'voting' or 'stacking' ")

        print(f"Training {ModelHelper._display_name(ensemble_model)}...")

        ensemble_model.fit(X_train, y_train)

    
        # === Threshold tuning and evaluation ===
        # Applies your ensemble model to the validation set.
        # Returns the predicted probabilities for the positive class (val_probs).
        # Ignores the default predictions (_) since we may tune them in the next step.
        # At this point, assume we are using the default threshold (e.g., 0.5).
        val_probs, _ = ModelHelper.apply_threshold(ensemble_model, X_val, threshold)

        # check if autotune threshold is enabled
        # if yes, then tune threshold between values from 0.1 to 0.9
        # comput the f1, recall and precision and pick the best one
        # update threshold variable with the best optimized value
        if auto_tune_threshold and val_probs is not None:
            print(f"Auto tuning for model: {ModelHelper._display_name(ensemble_model)}")
            threshold, best_score= ModelHelper.tune_threshold(val_probs, y_val, metric=tune_metric)
            print("Auto tuning complete")
        else:
            best_score = None
        # Now re-applies the optimized threshold:
        # val_preds: Final predictions on validation set
        # test_preds: Final predictions on test set
        # Also returns the probabilities (val_probs, test_probs) if we want ROC-AUC or calibration metrics.

        val_probs, val_preds = ModelHelper.apply_threshold(ensemble_model, X_val, threshold)
        test_probs, test_preds = ModelHelper.apply_threshold(ensemble_model, X_test, threshold)

        # Predict on val and test
        print(f"Evaluating for model: {ModelHelper._display_name(ensemble_model)}")

        val_metrics = ModelHelper.evaluate(y_val, val_preds, val_probs)

        print("Evaluation complete.")

        print(f"Testing for model: {ModelHelper._display_name(ensemble_model)}")

        test_metrics = ModelHelper.evaluate(y_test, test_preds, test_probs)

        print("Testing complete")

        
        print("Ensemble Validation Report:")
        print(classification_report(y_val, val_preds))
        print("Ensemble Test Report:")
        print(classification_report(y_test, test_preds))

        # Save model and metadata
        run_path = ModelHelper.create_model_run_folder(base_dir)
        model_path = os.path.join(run_path, "ensemble_model.pkl")

        metadata = {
            "model_type": "VotingClassifier",
            "base_models": list(models.keys()),
            "dataset": dataset_name,
            "threshold": threshold,
            "metrics": {k: float(v) for k, v in test_metrics.items()},
            "validation_metrics": {k: float(v) for k, v in val_metrics.items()},
            "hyperparameters": {name: model.get_params() for name, model in models.items()},
            "split_counts": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test)
            },
            "training_info": {
                "ensemble_type": ensemble_type,
                "voting_strategy": voting if ensemble_type == "voting" else None,
                "weights": weights if ensemble_type == "voting" else None,
                "final_estimator": ModelHelper._display_name(ensemble_model) if final_estimator else "LogisticRegression", # str(type(final_estimator).__name__) 
                "run_id": os.path.basename(run_path),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "voting": voting,
                "auto_tune_threshold": auto_tune_threshold,
                "tune_metric": tune_metric,
                "best_score_on_val": round(best_score, 4) if best_score else None
            },
            "environment": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "sklearn_version": sklearn.__version__,
                "libraries": {
                    "pandas": pd.__version__,
                    "seaborn": sns.__version__,
                    "matplotlib": plt.matplotlib.__version__,
                    "joblib": joblib.__version__
                }
            }
        }

        ModelHelper.save_model(ensemble_model, model_path, metadata)

        # Save confusion matrices
        dhelp.save_confusion_matrix(
            y_val, val_preds, "Ensemble Validation Confusion Matrix",
            os.path.join(run_path, "ensemble_val_confusion_matrix.png")
        )

        dhelp.save_confusion_matrix(
            y_test, test_preds, "Ensemble Test Confusion Matrix",
            os.path.join(run_path, "ensemble_test_confusion_matrix.png")
        )

        # Save feature importance plot (if applicable)
        dhelp.save_feature_importance_plot(ensemble_model, X_train.columns.tolist(), run_path, model_label="VotingClassifier")

        #save precision recall curve
        pr_curve_path = os.path.join(run_path, "ensemble_precision_recall_curve.png")
        dhelp.plot_precision_recall_curve(ensemble_model, y_val, val_probs, save_path=pr_curve_path, model_label="VotingClassifier")

        return ensemble_model, run_path, test_metrics

    @staticmethod    
    def build_model(model_key: str):
        """
        Return (pipeline, param_grid) for model_key in
        ["RF","XGB","LR","LGBM","PCA_LR"].
        Uses the leak-free preprocessors defined earlier.
        """
        # --- shared pieces from previous refactor ---
        numeric_cols = ["age_mid", "tumor_size_mid", "inv_nodes_mid", "age_x_tumor", "tumor_per_node"]
        nominal_cols = ["menopause", "node-caps", "irradiat"]
        freq_cols    = ["breast", "breast-quad"]


        pre_linear = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ])
        
        pre_tree = ColumnTransformer([
            # frequency-encode breast / breast-quad
            ("freq", FrequencyEncoder(columns=freq_cols, drop_original=True), freq_cols),
            # ordinal-encode menopause, node-caps, irradiat
            ("ord",  OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ),
                    nominal_cols),
            # numeric passthrough
            ("num",  "passthrough", numeric_cols),
        ])


        # ---- model-specific definitions ----
        if model_key == "RF":
            pipe = ImbPipeline([
                ("pre",   pre_tree),
                ("smote", SMOTE(random_state=42)),
                ("clf",   RandomForestClassifier(random_state=42))
            ])
            grid = {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 8, 12],
                "clf__min_samples_leaf": [1, 5],
                "clf__min_samples_split": [2, 5],
                "clf__max_features": ["sqrt"],
            }

        elif model_key == "XGB":
            pipe = ImbPipeline([
                ("pre",   pre_tree),
                ("smote", SMOTE(random_state=42)),
                ("clf",   XGBClassifier(
                    random_state=42,
                    #use_label_encoder=False,
                    eval_metric="logloss",
                    tree_method="hist"   # swap to "gpu_hist" if you *know* CUDA is available
                ))
            ])
            grid = {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.01, 0.1],
                "clf__max_depth": [3, 6, 8],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_lambda": [1, 5],
            }

        elif model_key == "LR":
            pipe = ImbPipeline([
                ("pre",   pre_linear),
                ("smote", SMOTE(random_state=42)),
                ("clf",   LogisticRegression(max_iter=2000,
                                            class_weight="balanced",
                                            solver="saga",
                                            random_state=42))
            ])
            grid = {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            }

        elif model_key == "PCA_LR":
            pipe = ImbPipeline([
                ("pre",   pre_linear),
                ("smote", SMOTE(random_state=42)),
                ("pca",   PCA()),
                ("clf",   LogisticRegression(max_iter=2000,
                                            class_weight="balanced",
                                            solver="saga",
                                            random_state=42))
            ])
            grid = {
                "pca__n_components": [5, 10],
                "clf__C": [0.1, 1.0, 10.0],
            }

        elif model_key == "LGBM":
            pipe = ImbPipeline([
                ("pre",   pre_tree),
                ("smote", SMOTE(random_state=42)),
                ("clf",   LGBMClassifier(
                    random_state=42,
                    force_row_wise=True,
                    verbose=-1
                ))
            ])
            grid = {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.01, 0.05],
                "clf__num_leaves": [15, 31],
                "clf__max_depth": [-1, 5, 10],
            }

        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        return pipe, grid