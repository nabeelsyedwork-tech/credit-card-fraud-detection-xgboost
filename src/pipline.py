import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('creditcard.csv')
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weight
weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Base XGBoost model
base_model = XGBClassifier(scale_pos_weight=weight, eval_metric='mlogloss')
base_model.fit(x_train, y_train)

# Evaluate base model
y_pred_base = base_model.predict(x_test)
print("Base Model")
print(confusion_matrix(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
x_train_new = selector.fit_transform(x_train, y_train)
x_test_new  = selector.transform(x_test)

# Define evaluation function for Bayesian Optimization
def xgb_evaluate(max_depth, learning_rate, n_estimators, scale_pos_weight, k):
    k = int(k)
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    scale_pos_weight = float(scale_pos_weight)

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        eval_metric='mlogloss'
    )

    scores = cross_val_score(model, x_train_new, y_train, cv=5, scoring='f1', n_jobs=-1)
    return scores.mean()

# Define hyperparameter bounds
pbounds = {
    'max_depth': (2, 6),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (10, 100),
    'scale_pos_weight': (1, weight),
    'k': (5, X.shape[1])
}

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=1
)

# Run optimization
optimizer.maximize(init_points=5, n_iter=10)

# Get best parameters
best_params = optimizer.max['params']

# Train final tuned model
final_model = XGBClassifier(
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    n_estimators=int(best_params['n_estimators']),
    scale_pos_weight=float(best_params['scale_pos_weight']),
    eval_metric='mlogloss',
)

final_model.fit(x_train_new, y_train)

# Evaluate tuned model
y_pred_tuned = final_model.predict(x_test_new)
print("Tuned Model")
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))
