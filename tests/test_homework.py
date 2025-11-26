"""Autograding script."""


def load_data():
    import pandas as pd
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "files" / "input" / "heart_disease.csv"
    dataset = pd.read_csv(csv_path)
    y = dataset.pop("target")
    x = dataset.copy()
    x["thal"] = x["thal"].map(
        lambda x: "normal" if x not in ["fixed", "fixed", "reversible"] else x
    )

    return x, y


def load_estimator():

    import pickle
    from pathlib import Path

    estimator_path = Path(__file__).resolve().parent.parent / "homework" / "estimator.pickle"
    if not estimator_path.exists():
        return None
    with open(estimator_path, "rb") as file:
        estimator = pickle.load(file)

    return estimator


def test_01():

    from sklearn.metrics import accuracy_score

    x, y = load_data()
    estimator = load_estimator()

    accuracy = accuracy_score(
        y_true=y,
        y_pred=estimator.predict(x),
    )

    assert accuracy > 0.85
