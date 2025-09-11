import os
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class DataHelper:

    """
    A helper class to help with data modificiations, charts, fixes and more
    """
    @staticmethod
    def generate_splits(
        X, y,
        split_strategy = "60/20/20",
        # test_size=0.2,
        # val_size=0.25,
        random_state=42
    ):
        """
        Generates train/val/test splits based on the chosen strategy.

        Parameters:
        - X: Features (must be a DataFrame)
        - y: Labels (Series or array-like)
        - split_strategy: One of '60/20/20', '80/10/10', or '70/15/15'
        - random_state: Seed for reproducibility

        Returns:
        - Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with column names.")
        

        # Set proportions based on strategy
        if split_strategy == "60/20/20":
            test_size = 0.2
            val_size = 0.25  # 25% of 80% = 20%
        elif split_strategy == "80/10/10":
            test_size = 0.1
            val_size = 0.1111  # ~11.11% of 90% = 10%
        elif split_strategy == "70/15/15":
            test_size = 0.15
            val_size = 0.1765  # ~17.65% of 85% = 15%
        else:
            raise ValueError("Unsupported split strategy. Choose '60/20/20', '80/10/10', or '70/15/15'.")

        # Step 1: Test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Step 2: Validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
        )

        # Wrap everything as DataFrames again to preserve structure
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_val = pd.DataFrame(X_val, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        y_train = pd.Series(y_train, name="Class")
        y_val = pd.Series(y_val, name="Class")
        y_test = pd.Series(y_test, name="Class")

        # Logging
        print(f"\n=== Split Strategy: {split_strategy} ===")
        print(f"Train set:     {len(X_train):,} samples")
        print(f"Validation set:{len(X_val):,} samples")
        print(f"Test set:      {len(X_test):,} samples")
        print(f"Total:         {len(X_train) + len(X_val) + len(X_test):,} samples")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }

    @staticmethod
    def generate_kfold_splits(X, y, n_splits=5, shuffle=True, random_state=42):
        """
        Generates stratified K-fold training and validation splits.

        Parameters:
        - X: Feature matrix (array or DataFrame)
        - y: Target vector (array or Series)
        - n_splits: Number of folds (default is 5)
        - shuffle: Whether to shuffle the data before splitting
        - random_state: Random seed for reproducibility

        Yields:
        - A generator of dictionaries with keys: X_train, X_val, y_train, y_val
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]


            yield {
                "X_train": X_train,
                "X_val": X_val,
                "y_train": y_train,
                "y_val": y_val
            }

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance", save_path=None):
        """
        Plots feature importances for tree-based models.

        Parameters:
        - model: trained XGB or RF model (must have .feature_importances_)
        - feature_names: list of feature names in the same order as training
        - top_n: number of top features to display
        - title: title of the plot
        - save_path: if provided, saves the plot as PNG
        """

        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not support feature importances.")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, hue="Feature", palette="viridis", legend=False)
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f" Saved feature importance plot to: {save_path}")
        else:
            plt.show()

    @staticmethod
    def missing_summary(df):
        """
        Prints a summary of missing values in the DataFrame.
        """
        missing_counts = df.isna().sum()
        total_rows = len(df)

        print(f"\n Missing Value Summary:")
        print(f"- Total rows: {total_rows}")
        print(f"- Columns with missing values:")
        print(missing_counts[missing_counts > 0])

        print(f"\n Total rows with any missing value: {df.isna().any(axis=1).sum()}")

    @staticmethod
    def save_confusion_matrix(y_true, y_pred, title, path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def range_to_midpoint(series):
        return series.str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1)
    
    @staticmethod
    def save_feature_importance_plot(model, feature_names, run_path, model_label=None):
        """
        Generates and saves a feature importance plot for a model.

        Parameters:
        - model: The trained model object (must support feature importance extraction).
        - feature_names: List of feature names used in training.
        - run_path: Directory path where the plot will be saved.
        - model_label: Optional custom label to use for the saved file and plot title.
        """
        
        try:
            model_name = model_label if model_label else type(model).__name__
            save_path = os.path.join(run_path, f"{model_name}_feature_importance.png")
            DataHelper.plot_feature_importance(
                model=model,
                feature_names=feature_names,
                save_path=save_path
            )
        except ValueError as e:
            print(f"[Info] Feature importance not available for {type(model).__name__}: {str(e)}")

    @staticmethod
    def plot_precision_recall_curve(model, y_true, probs, save_path=None,model_label=None):
        """
        Plots Precision and Recall scores against decision thresholds.

        Parameters:
        - model: The model used for prediction (used for labeling only).
        - y_true: True labels for the dataset.
        - probs: Predicted probabilities from the model.
        - save_path: Optional path to save the plot image.
        - model_label: Optional custom label to use in the plot title.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, probs)

        model_name = model_label if model_label else type(model).__name__

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precision[:-1], label="Precision")
        plt.plot(thresholds, recall[:-1], label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"{ model_name }Precision-Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()