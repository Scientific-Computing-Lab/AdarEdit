import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import shap  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  
from sklearn.preprocessing import StandardScaler  
import xgboost as xgb  
import warnings  
from tqdm import tqdm
import argparse
from pathlib import Path
import pickle



  
warnings.filterwarnings("ignore")  
  
# Load and preprocess the dataset  
def load_data(csv_path):  
    """Load the dataset from a CSV file and preprocess it."""  
    print("Step 1: Loading data...")  
    with tqdm(total=100, desc="Loading data", unit="%", leave=True) as pbar:  
        df = pd.read_csv(csv_path)  
        pbar.update(50)  # Simulate progress  
        df.fillna(0, inplace=True)  # Handle missing values  
        pbar.update(50)  # Complete progress  
        X = df.loc[:, "pos_-600":"pos_600"]  # Extract attention values as features  
        y = df["ground_truth"]  # Extract ground truth as target labels  
    return X, y, df  
  
# Calculate baseline metrics  
def calculate_baseline_metrics(df):  
    """Calculate baseline accuracy metrics based on model_prediction and ground_truth."""  
    print("\nStep 0: Calculating baseline accuracy metrics...")  
    baseline_predictions = df["model_prediction"]  
    ground_truth = df["ground_truth"]  
  
    accuracy = accuracy_score(ground_truth, baseline_predictions)  
    precision = precision_score(ground_truth, baseline_predictions)  
    recall = recall_score(ground_truth, baseline_predictions)  
    f1 = f1_score(ground_truth, baseline_predictions)  
  
    print(f"\nBaseline Metrics:")  
    print(f"Accuracy: {accuracy:.4f}")  
    print(f"Precision: {precision:.4f}")  
    print(f"Recall: {recall:.4f}")  
    print(f"F1-Score: {f1:.4f}")  
    print("Classification Report:")  
    print(classification_report(ground_truth, baseline_predictions))  
  
    return accuracy, precision, recall, f1  
  
# Perform grid search for hyperparameter tuning  
def grid_search_xgboost(dtrain, dtest, y_test, param_grid):  
    print("\nPerforming hyperparameter tuning...")  
    best_model = None  
    best_params = None  
    best_f1 = -1  
  
    for max_depth in param_grid['max_depth']:  
        for learning_rate in param_grid['learning_rate']:  
            for n_estimators in param_grid['n_estimators']:  
                for colsample_bytree in param_grid['colsample_bytree']:  
                    for gamma in param_grid['gamma']:  
                        for min_child_weight in param_grid['min_child_weight']:  
                            params = {  
                                'max_depth': max_depth,  
                                'learning_rate': learning_rate,  
                                'n_estimators': n_estimators,  
                                'colsample_bytree': colsample_bytree,  
                                'gamma': gamma,  
                                'min_child_weight': min_child_weight,  
                                'objective': 'binary:logistic',  
                                'eval_metric': 'logloss',  
                                'tree_method': 'gpu_hist',  # Use GPU for training  
                                'predictor': 'gpu_predictor'  # Use GPU for prediction  
                            }  
  
                            # Train the model  
                            model = xgb.train(params, dtrain, num_boost_round=n_estimators)  
  
                            # Evaluate the model  
                            y_pred = (model.predict(dtest) > 0.5).astype(int)  
                            f1 = f1_score(y_test, y_pred)  
  
                            if f1 > best_f1:  
                                best_f1 = f1  
                                best_model = model  
                                best_params = params  
  
    print(f"\nBest Parameters: {best_params}")  
    print(f"Best F1-Score: {best_f1:.4f}")  
    return best_model, best_params  
  
# Train and evaluate XGBoost with SHAP explanations  
def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, original_feature_names, top_n_features,output_dir):  
    print("\nStep 3: Training and evaluating XGBoost...")  
  
    # Standardize the features  
    scaler = StandardScaler()  
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)  
  
    # Move data to GPU using xgboost.DMatrix  
    print("\nMoving data to GPU...")  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dtest = xgb.DMatrix(X_test, label=y_test)  
  
    # Define the larger parameter grid  
    param_grid = {  
        'max_depth': [3, 5, 7, 10],  
        'learning_rate': [0.01, 0.05, 0.1],  
        'n_estimators': [50, 100, 200],  
        'colsample_bytree': [0.8, 1.0],  
        'gamma': [0, 0.1, 0.2],  
        'min_child_weight': [1, 3, 5]  
    }  
  
    # Perform grid search for hyperparameter tuning  
    best_model, best_params = grid_search_xgboost(dtrain, dtest, y_test, param_grid)  
  
    # Evaluate the best model  
    y_pred = (best_model.predict(dtest) > 0.5).astype(int)  
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  
  
    print(f"\nOriginal Model - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")  
    print("Classification Report:")  
    print(classification_report(y_test, y_pred))  
  
    # Compute SHAP explanations  
    print("\nStep 4: Computing SHAP explanations...")  
    explainer = shap.TreeExplainer(best_model)  
    shap_values = explainer.shap_values(X_test)  
  
    # Identify top N features based on mean absolute SHAP values  
    print(f"\nIdentifying top {top_n_features} features...")  
    mean_shap_values = np.abs(shap_values).mean(axis=0)  
    top_indices = np.argsort(mean_shap_values)[-top_n_features:][::-1]  
    top_features = [original_feature_names[i] for i in top_indices]  
  
    print("\nTop Features (Original Model):")  
    for i, feature in enumerate(top_features):  
        print(f"{i+1}. {feature} (Mean SHAP Value: {mean_shap_values[top_indices[i]]:.4f})")  
  
    # Save SHAP plot for original model  
    plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})
    shap.summary_plot(shap_values, X_test, feature_names=original_feature_names, max_display=top_n_features, show=False)  
    plt.title(f"Top {top_n_features} SHAP Feature Importance for Original Model")  
    plt.savefig(output_dir /"shap_top20_XGBoost.png")  
    plt.close() 

    with open(output_dir / "shap_data_Original_Model.pkl", "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "X_test": X_test,
            "feature_names": original_feature_names
        }, f)
   
  
    # Retrain the model using only the top features  
    print("\nRetraining the model using top features...")  
    X_train_top = X_train[:, top_indices]  
    X_test_top = X_test[:, top_indices]  
  
    dtrain_top = xgb.DMatrix(X_train_top, label=y_train)  
    dtest_top = xgb.DMatrix(X_test_top, label=y_test)  
  
    best_model_top, best_params_top = grid_search_xgboost(dtrain_top, dtest_top, y_test, param_grid)  
  
    # Verify that the features used for retraining match the top features  
    print("\nValidating feature list consistency...")  
    used_features = [original_feature_names[i] for i in top_indices]  
    assert used_features == top_features, "Feature lists before and after retraining do not match!"  
    print("Feature list consistency validated successfully.")  
  
    # Evaluate the retrained model  
    y_pred_top = (best_model_top.predict(dtest_top) > 0.5).astype(int)  
    accuracy_top = accuracy_score(y_test, y_pred_top)  
    precision_top = precision_score(y_test, y_pred_top)  
    recall_top = recall_score(y_test, y_pred_top)  
    f1_top = f1_score(y_test, y_pred_top)  
  
    print(f"\nRetrained Model - Test Accuracy: {accuracy_top:.4f}, Precision: {precision_top:.4f}, Recall: {recall_top:.4f}, F1-Score: {f1_top:.4f}")  
    print("Classification Report (Retrained Model):")  
    print(classification_report(y_test, y_pred_top))  
  
    # Compute SHAP explanations for the retrained model  
    print("\nStep 5: Computing SHAP explanations for the retrained model...")  
    explainer_top = shap.TreeExplainer(best_model_top)  
    shap_values_top = explainer_top.shap_values(X_test_top)  
  
    # Display SHAP feature importance values before saving the plot  
    print("\nSHAP Feature Importance (Retrained Model):")  
    for i, feature in enumerate(top_features):  
        print(f"{i+1}. {feature} (Mean SHAP Value: {np.abs(shap_values_top).mean(axis=0)[i]:.4f})")  
  
    # Save SHAP plot for retrained model  
    plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})
    shap.summary_plot(shap_values_top, X_test_top, feature_names=top_features, max_display=top_n_features, show=False)  
    plt.title(f"Top {top_n_features} SHAP Feature Importance for Retrained Model")  
    plt.savefig(output_dir /"shap_top20_Retrained_XGBoost.png")  
    plt.close()  


    with open(output_dir / "shap_data_Retrained_Model.pkl", "wb") as f:
        pickle.dump({
            "shap_values": shap_values_top,
            "X_test": X_test_top,
            "feature_names": top_features
        }, f)
  
# Main script  
if __name__ == "__main__":  
    top_n_features = 20  # Number of top features to select  

    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_csv", type=str, required=True)
    args = parser.parse_args()
    csv_file_path = args.attention_csv


    output_dir = Path(csv_file_path).parent  # This will be the experiment folder

    # Step 1: Loading data  
    X, y, df = load_data(csv_file_path)  
  
    # Step 0: Calculate baseline metrics  
    baseline_accuracy, baseline_precision, baseline_recall, baseline_f1 = calculate_baseline_metrics(df)  
  
    # Step 2: Preparing data for XGBoost  
    print("\nStep 2: Preparing data for XGBoost...")  
    with tqdm(total=100, desc="Preparing data", unit="%", leave=True) as pbar:  
        # Split the data into training and test sets (80% train, 20% test)  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  
        pbar.update(100)  # Complete progress  
  
    # Save original feature names for SHAP plots  
    original_feature_names = X.columns.tolist()  
  
    # Step 3: Train and evaluate XGBoost  
    train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, original_feature_names, top_n_features,output_dir)  
