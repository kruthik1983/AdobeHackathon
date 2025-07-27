import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = [
    'font_size', 
    'is_bold', 
    'is_centered', 
    'space_above', 
    'x0', 
    'word_count'
]

def train_and_save_model(csv_path: str, model_dir: str):
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
    df = df.dropna(subset=['level'])

    X = df[MODEL_FEATURES].to_dict(orient='records')
    y = df['level']

    vectorizer = DictVectorizer(sparse=False)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

    # Define models and their grids
    models_and_grids = [
        (RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }),
        (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 7]
        }),
        (make_pipeline(StandardScaler(), LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000, solver='lbfgs')), {
        'logisticregression__C': [1.0, 10.0]
        }),
    ]

    best_models = []
    best_scores = []
    print("Starting model fine-tuning with GridSearchCV... (This may take a few minutes)")
    for model, grid in models_and_grids:
        grid_search = GridSearchCV(model, grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        best_models.append(best_model)
        best_scores.append(acc)
        print(f"{type(best_model).__name__} best accuracy: {acc:.2%} | Params: {grid_search.best_params_}")

    # Consensus (majority vote) using VotingClassifier
    estimators = [(f"model_{i}", m) for i, m in enumerate(best_models)]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred_consensus = voting_clf.predict(X_test)
    consensus_acc = accuracy_score(y_test, y_pred_consensus)
    print(f"\nConsensus (VotingClassifier) accuracy: {consensus_acc:.2%}")

    # Save the best individual model and the consensus model
    os.makedirs(model_dir, exist_ok=True)
    # Save vectorizer and consensus model
    joblib.dump({'vectorizer': vectorizer, 'model': voting_clf, 'classes': voting_clf.classes_}, os.path.join(model_dir, 'pdf_heading_model.joblib'))
    print(f"\nConsensus model successfully trained and saved to '{os.path.join(model_dir, 'pdf_heading_model.joblib')}'")

    # Optionally, save the best individual model too
    best_idx = int(np.argmax(best_scores))
    joblib.dump({'vectorizer': vectorizer, 'model': best_models[best_idx], 'classes': best_models[best_idx].classes_}, os.path.join(model_dir, 'pdf_heading_best_individual_model.joblib'))
    print(f"Best individual model ({type(best_models[best_idx]).__name__}) saved to '{os.path.join(model_dir, 'pdf_heading_best_individual_model.joblib')}'")

def classify_headings(feature_lines: list, model_dir: str):
    """
    Classifies text lines using the fine-tuned, pre-trained model.
    """
    if not feature_lines:
        return "", []

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
