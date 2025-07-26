import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
# --- CHANGE: Imported tools for model evaluation and hyperparameter tuning ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

MODEL_FEATURES = [
    'font_size', 
    'is_bold', 
    'is_centered', 
    'space_above', 
    'x0', 
    'word_count'
]

def train_and_save_model(csv_path: str, model_dir: str):
    """
    Loads the dataset, fine-tunes a RandomForest model using GridSearchCV to 
    achieve higher accuracy, evaluates it, and saves the best model.
    """
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at '{csv_path}'. Please ensure the file exists.")
        return

    print("Preparing data for fine-tuning...")
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[MODEL_FEATURES] = df[MODEL_FEATURES].fillna(0)
    
    X = df[MODEL_FEATURES].to_dict(orient='records')
    y = df['level']

    vectorizer = DictVectorizer(sparse=False)
    X_vec = vectorizer.fit_transform(X)
    
    # Split data into training and testing sets to evaluate accuracy properly
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

    # --- CHANGE: Define a grid of hyperparameters to search through ---
    # This grid explores different combinations to find the most accurate model.
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    rfc = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # --- CHANGE: Use GridSearchCV to find the best model ---
    # It will test combinations from param_grid using 5-fold cross-validation.
    print("Starting model fine-tuning with GridSearchCV... (This may take a few minutes)")
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("\n--- Fine-Tuning Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Get the best model from the search
    best_model = grid_search.best_estimator_

    # --- CHANGE: Evaluate the best model on the test set ---
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.2%}")

    if accuracy < 0.90:
        print("NOTE: Target accuracy of 90% was not met. Consider expanding the dataset or adding more features.")

    model_payload = {
        'model': best_model,
        'vectorizer': vectorizer,
        'classes': best_model.classes_
    }

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'pdf_heading_model.joblib')
    joblib.dump(model_payload, model_path)
    print(f"\nBest model successfully trained and saved to '{model_path}'")


def classify_headings(feature_lines: list, model_dir: str):
    """
    Classifies text lines using the fine-tuned, pre-trained model.
    """
    if not feature_lines:
        return "Untitled Document", []

    model_path = os.path.join(model_dir, 'pdf_heading_model.joblib')
    
    try:
        model_payload = joblib.load(model_path)
        model = model_payload['model']
        vectorizer = model_payload['vectorizer']
        class_names = model_payload['classes']
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'.")
        print("Please run 'python heading_classifier_ml.py' to train the model.")
        return "Error: Model not found", []

    features_to_predict = [{k: line.get(k, 0) for k in MODEL_FEATURES} for line in feature_lines]
    X_new_vec = vectorizer.transform(features_to_predict)
    
    predictions = model.predict(X_new_vec)
    probabilities = model.predict_proba(X_new_vec)
    
    title_class_index = np.where(class_names == 'Title')[0][0] if 'Title' in class_names else -1

    title = ""
    outline = []
    
    best_title_prob = 0.0
    best_title_index = -1

    if title_class_index != -1:
        for i, prob_dist in enumerate(probabilities):
            title_prob = prob_dist[title_class_index]
            if title_prob > best_title_prob and feature_lines[i]['page_number'] == 1:
                best_title_prob = title_prob
                best_title_index = i

    if best_title_index != -1 and best_title_prob > 0.70:
        title = feature_lines[best_title_index]['text'].strip()
    
    for i, pred in enumerate(predictions):
        if i == best_title_index and best_title_prob > 0.70:
            continue
        if pred in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            outline.append({
                "level": pred,
                "text": feature_lines[i]['text'].strip(),
                "page": feature_lines[i]['page_number']
            })
            
    return title, outline

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'dataset', 'pdf_analyzer.csv')
    DATASET_CSV_PATH = path
    MODEL_SAVE_DIR = 'model'
    
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"Dataset not found at {DATASET_CSV_PATH}. Please ensure it exists.")
    else:
        train_and_save_model(DATASET_CSV_PATH, MODEL_SAVE_DIR)
