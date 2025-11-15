import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def prepare_train(numerical_features, categorical_features):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    model = DecisionTreeClassifier(random_state=42)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    return clf

def evaluation(X_test, y_test, clf):
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # for ROC AUC

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("ROC AUC:", roc_auc_score(y_test, y_proba))


if __name__ == "__main__":
    df = pd.read_csv('../data/secondary_data.csv', delimiter=';')

    numerical = df.select_dtypes(include=["float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    categorical.remove("class")  # Remove target variable

    # Remove columns with lots of missing values
    cols = ['stem-root', 'veil-type', 'veil-color', 'spore-print-color']
    categorical = list(set(categorical) - set(cols))
    df.drop(cols, inplace=True, axis=1)
    df.fillna("unknown", inplace=True)

    X = df[numerical + categorical]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = prepare_train(numerical, categorical)
    clf.fit(X_train, y_train)

    print("Train")
    evaluation(X_train, y_train, clf)

    print("Test")
    evaluation(X_test, y_test, clf)

    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(clf, f)


