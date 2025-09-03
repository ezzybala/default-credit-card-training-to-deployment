import os
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Azure ML imports
from azureml.core import Run, Workspace, Model

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model_output", type=str, help="output")
    
    args = parser.parse_args()

    # Get Azure ML run context
    run = Run.get_context()
    ws = run.experiment.workspace

    ###################
    # <prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)

    credit_df = pd.read_csv(args.data, header=1, index_col=0)

    # log dataset info
    run.log("num_samples", credit_df.shape[0])
    run.log("num_features", credit_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )
    ####################
    # </prepare the data>
    ####################

    ##################
    # <train the model>
    ##################
    y_train = train_df.pop("default payment next month")
    X_train = train_df.values

    y_test = test_df.pop("default payment next month")
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # log evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    run.log("test_accuracy", acc)
    print(classification_report(y_test, y_pred))
    ###################
    # </train the model>
    ###################

    ##########################
    # <save and register model>
    ##########################
    print("Saving and registering the model...")

    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "trained_model.pkl")

    # save with joblib
    joblib.dump(clf, model_path)

    # register model in Azure ML
    Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=args.registered_model_name,
        tags={"framework": "scikit-learn", "type": "classification"},
        description="Gradient Boosting Classifier"
    )
    ###########################
    # </save and register model>
    ###########################

if __name__ == "__main__":
    main()
