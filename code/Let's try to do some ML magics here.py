from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Additional imports for alternative feature importance methods:
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import RFE
import shap
import lime
import lime.lime_tabular

# Try importing TabNet (a new deep learning model with built‐in interpretability)
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("TabNetRegressor not available. To use TabNet, install pytorch-tabnet.")

# ==================== Data Processing ====================
class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""

    def __init__(self, filename: str):
        self.data = pd.read_csv(filename)

    def display_missing_info(self):
        """Prints columns with missing values and counts complete rows."""
        missing_values = self.data.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        print("Missing Values in Columns:")
        print(missing_columns.to_string())

        complete_rows = self.data.dropna().shape[0]
        print(f"\nNumber of rows with no missing values: {complete_rows}")

    def clean_data(self):
        """Drops specific columns and filters rows so that all columns (except 'number_of_data_centers')
        are not empty."""
        # Drop unnecessary columns
        self.data = self.data.drop([
            'average_sums', 'sums_sum', 'avg_durations_hrs', 'counts_of_outage',
            'avg_carbon_emission', 'avg_power_usage', 'CI90LB04_2023', 'CI90LB04P_2023',
            'CI90UB04P_2023', 'CI90UB04_2023', 'PCTPOV04_2023', 'POV04_2023'
        ], axis=1)

        # Filter rows where every column except 'number_of_data_centers' is not missing
        cols_to_check = [col for col in self.data.columns if col != 'number_of_data_centers']
        self.data = self.data.dropna(subset=cols_to_check)
        print(f"\nNumber of rows after cleaning: {self.data.shape[0]}")

    def get_features_and_target(self):
        """Separates and returns the features (X) and target (y)."""
        y = self.data['sum_duration_hrs']
        X = self.data.drop(['sum_duration_hrs'], axis=1)
        return X, y

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values, then standardize and scale to [0, 1] (single pipeline, fit once)."""
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('standard_scaler', StandardScaler()),
            ('minmax_scaler', MinMaxScaler()),
        ])
        X_processed = pipeline.fit_transform(X)
        X_processed_df = pd.DataFrame(X_processed, columns=X.columns, index=X.index)

        print("\nProcessed Feature Data (first 5 rows):")
        print(X_processed_df.head())
        return X_processed_df


# ==================== Baseline Model Development ====================
class ModelTrainer:
    """Handles model training, evaluation, and data splitting."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.model = Lasso(alpha=0.1, random_state=1, max_iter=10000)


    def split_data(self, test_size: float = 0.2, random_state: int = 1):
        """Splits data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train(self):
        """Trains the LASSO regression model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluates the model on the test set and prints error metrics."""
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print("\nLASSO Regression Baseline Model")
        print("Mean Absolute Error (MAE):", mae)
        print("Root Mean Squared Error (RMSE):", rmse)
        return mae, rmse


# ==================== Feature Importance and Interpretation ====================
class FeatureImportanceInterpreter:
    """Interprets and displays feature importance based on model coefficients."""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def lasso_importance(self):
        """Displays importance based on LASSO coefficients."""
        coefficients = self.model.coef_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients
        })
        importance_df['abs_coef'] = importance_df['coefficient'].abs()
        importance_df.sort_values(by='abs_coef', ascending=False, inplace=True)
        print("\nLASSO Feature Importance:")
        print(importance_df.to_string())
        return importance_df


# ==================== Advanced & Alternative Feature Importance ====================
class AdvancedFeatureImportance:
    """Provides alternative and advanced methods to evaluate feature importance."""

    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def random_forest_importance(self):
        """Uses Random Forest to compute feature importances."""
        rf = RandomForestRegressor(random_state=1)
        rf.fit(self.X_train, self.y_train)
        importances = rf.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        print("\nRandom Forest Feature Importance:")
        print(importance_df.to_string())
        return importance_df

    def permutation_importance(self, estimator):
        """Calculates permutation importance on the provided estimator."""
        result = permutation_importance(estimator, self.X_test, self.y_test, n_repeats=10, random_state=1)
        importance_df = pd.DataFrame({
            'feature': self.X_test.columns,
            'permutation_importance': result.importances_mean
        }).sort_values(by='permutation_importance', ascending=False)
        print("\nPermutation Feature Importance:")
        print(importance_df.to_string())
        return importance_df

    def shap_importance(self, estimator):
        """Uses SHAP to compute and display feature importances."""
        # Pass the training data as a NumPy array and supply the feature names explicitly.
        explainer = shap.Explainer(estimator, self.X_train.values, feature_names=self.X_train.columns)
        shap_values = explainer(self.X_test)
        print("\nGenerating SHAP summary plot (this will display a plot)...")
        shap.summary_plot(shap_values, self.X_test, show=False)
        return shap_values


    def lime_explanation(self, estimator, instance_idx=0, num_features=10):
        """Uses LIME to explain a single prediction from the estimator."""
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns,
            mode='regression'
        )
        exp = explainer.explain_instance(self.X_test.iloc[instance_idx].values, estimator.predict, num_features=num_features)
        print("\nLIME Explanation for instance {}:".format(instance_idx))
        try:
            exp.show_in_notebook(show_all=False)
        except Exception:
            print(exp.as_list())
        return exp

    def rfe_feature_selection(self, n_features_to_select=5):
        """Uses Recursive Feature Elimination (RFE) with LASSO to select important features."""
        rfe_selector = RFE(estimator=Lasso(alpha=0.1, random_state=1), n_features_to_select=n_features_to_select)
        rfe_selector.fit(self.X_train, self.y_train)
        selected_features = self.X_train.columns[rfe_selector.support_]
        print(f"\nSelected features via RFE (n={n_features_to_select}):")
        print(selected_features.tolist())
        return selected_features

    def partial_dependence_plot(self, estimator, features_to_plot=None):
        """Plots Partial Dependence Plots for selected features."""
        if features_to_plot is None:
            # Default: Plot PDP for the first two features
            features_to_plot = [0, 1]
        print("\nDisplaying Partial Dependence Plots...")
        fig, ax = plt.subplots(figsize=(10, 5))
        PartialDependenceDisplay.from_estimator(estimator, self.X_train, features_to_plot, ax=ax)
        plt.show()

    def tabnet_feature_importance(self):
        """Trains a TabNet model (if available) and prints its built-in feature importances."""
        if not TABNET_AVAILABLE:
            print("TabNet is not installed. Skipping TabNet feature importance.")
            return None

        # TabNet requires numpy arrays; here we simply fit and extract the feature importance via its attention mechanism.
        tabnet = TabNetRegressor(verbose=0, seed=1)
        X_train_np = self.X_train.values
        y_train_np = self.y_train.values.reshape(-1, 1)
        tabnet.fit(X_train_np, y_train_np, max_epochs=50, patience=5, batch_size=256, virtual_batch_size=128)

        # TabNet provides feature importances via its explain() method:
        explanations, masks = tabnet.explain(X_train_np)
        # Compute average feature importance over all training examples:
        avg_importance = np.mean(masks, axis=0).squeeze()
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'tabnet_importance': avg_importance
        }).sort_values(by='tabnet_importance', ascending=False)
        print("\nTabNet Feature Importance:")
        print(importance_df.to_string())
        return importance_df


# ==================== Main Execution ====================
def main():
    # Data Processing (repo root = directory containing this script)
    root = Path(__file__).resolve().parent
    processor = DataProcessor(root / 'Merged for ML.csv')
    processor.display_missing_info()
    processor.clean_data()
    X, y = processor.get_features_and_target()
    X_processed = processor.preprocess_features(X)

    # Baseline Model Development
    trainer = ModelTrainer(X_processed, y)
    trainer.split_data()
    trainer.train()
    trainer.evaluate()

    # LASSO-based Feature Importance
    interpreter = FeatureImportanceInterpreter(trainer.model, X_processed.columns.tolist())
    interpreter.lasso_importance()

    # Advanced Feature Importance Methods
    adv_importance = AdvancedFeatureImportance(trainer.X_train, trainer.X_test, trainer.y_train, trainer.y_test)

    # 1. Random Forest Importance
    rf_imp = adv_importance.random_forest_importance()

    # 2. Permutation Importance (using the trained LASSO model as an example)
    perm_imp = adv_importance.permutation_importance(trainer.model)

    # 3. SHAP Importance (for LASSO or you can use the RF model; here we use LASSO)
    shap_vals = adv_importance.shap_importance(trainer.model)

    # 4. LIME Explanation (explain the first instance in test set)
    lime_exp = adv_importance.lime_explanation(trainer.model, instance_idx=0)

    # 5. Recursive Feature Elimination (RFE)
    selected_features = adv_importance.rfe_feature_selection(n_features_to_select=5)

    # 6. Partial Dependence Plot (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        adv_importance.partial_dependence_plot(trainer.model, features_to_plot=[0, 1])
    except ImportError:
        print("matplotlib is not installed. Skipping PDP.")

    # 7. New Model: TabNet Feature Importance (if available)
    tabnet_imp = adv_importance.tabnet_feature_importance()


if __name__ == "__main__":
    main()
