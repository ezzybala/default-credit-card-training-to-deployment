#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()


# In[ ]:


# Handle to the workspace
# from azure.ai.ml import MLClient

# Authentication package
# from azure.identity import InteractiveBrowserCredential
# credential = InteractiveBrowserCredential()


# In[2]:


# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="cb51c13f-04b1-4395-8005-e4f9f2b5e397",
    resource_group_name="mlresources",
    workspace_name="MLAssessment",
)


# In[7]:


import os

dependencies_dir = "../dependencies"
os.makedirs(dependencies_dir, exist_ok=True)


# In[8]:


get_ipython().run_cell_magic('writefile', '{dependencies_dir}/conda.yaml', 'name: model-env\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.8\n  - numpy=1.21.2\n  - pip=21.2.4\n  - scikit-learn=0.24.2\n  - scipy=1.7.1\n  - pandas>=1.1,<1.2\n  - pip:\n    - inference-schema[numpy-support]==1.3.0\n    - xlrd==2.0.1\n    - mlflow== 1.26.1\n    - azureml-mlflow==1.42.0\n    - psutil>=5.8,<5.9\n    - tqdm>=4.59,<4.60\n    - ipykernel~=6.0\n    - matplotlib\n')


# In[9]:


from azure.ai.ml.entities import Environment

custom_env_name = "aml-scikit-learnv2"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)


# In[12]:


train_src_dir = "../src"
os.makedirs(train_dir, exist_ok=True)


# In[13]:


get_ipython().run_cell_magic('writefile', '{train_src_dir}/main.py', 'import os\nimport argparse\nimport pandas as pd\nimport mlflow\nimport mlflow.sklearn\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import classification_report\nfrom sklearn.model_selection import train_test_split\n\ndef main():\n    """Main function of the script."""\n\n    # input and output arguments\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--data", type=str, help="path to input data")\n    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)\n    parser.add_argument("--n_estimators", required=False, default=100, type=int)\n    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)\n    parser.add_argument("--registered_model_name", type=str, help="model name")\n    args = parser.parse_args()\n   \n    # Start Logging\n    mlflow.start_run()\n\n    # enable autologging\n    mlflow.sklearn.autolog()\n\n    ###################\n    #<prepare the data>\n    ###################\n    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))\n\n    print("input data:", args.data)\n    \n    credit_df = pd.read_excel(args.data, header=1, index_col=0)\n\n    mlflow.log_metric("num_samples", credit_df.shape[0])\n    mlflow.log_metric("num_features", credit_df.shape[1] - 1)\n\n    train_df, test_df = train_test_split(\n        credit_df,\n        test_size=args.test_train_ratio,\n    )\n    ####################\n    #</prepare the data>\n    ####################\n\n    ##################\n    #<train the model>\n    ##################\n    # Extracting the label column\n    y_train = train_df.pop("default payment next month")\n\n    # convert the dataframe values to array\n    X_train = train_df.values\n\n    # Extracting the label column\n    y_test = test_df.pop("default payment next month")\n\n    # convert the dataframe values to array\n    X_test = test_df.values\n\n    print(f"Training with data of shape {X_train.shape}")\n\n    clf = GradientBoostingClassifier(\n        n_estimators=args.n_estimators, learning_rate=args.learning_rate\n    )\n    clf.fit(X_train, y_train)\n\n    y_pred = clf.predict(X_test)\n\n    print(classification_report(y_test, y_pred))\n    ###################\n    #</train the model>\n    ###################\n\n    ##########################\n    #<save and register model>\n    ##########################\n    # Registering the model to the workspace\n    print("Registering the model via MLFlow")\n    mlflow.sklearn.log_model(\n        sk_model=clf,\n        registered_model_name=args.registered_model_name,\n        artifact_path=args.registered_model_name,\n    )\n\n    # Saving the model to a file\n    mlflow.sklearn.save_model(\n        sk_model=clf,\n        path=os.path.join(args.registered_model_name, "trained_model"),\n    )\n    ###########################\n    #</save and register model>\n    ###########################\n    \n    # Stop Logging\n    mlflow.end_run()\n\nif __name__ == "__main__":\n    main()\n')


# In[3]:


# Get data asset
data_asset = ml_client.data.get(name="default_credit_card_dataset", version="1")
print(data_asset.path)


# In[4]:


from azure.ai.ml import command
from azure.ai.ml import Input, Output

registered_model_name = "credit_defaults_model"

train_step = command(
    inputs=dict(
        data=Input(
            type="uri_file",
            path=data_asset.path,
        ),
        test_train_ratio=0.2,
        learning_rate=0.25,
        registered_model_name=registered_model_name,
    ),
    outputs=dict(
        model_output=Output(type="uri_folder")  # <-- define output
    ),
    code="../src/",  # location of source code
    command="python main2.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}} --model_output ${{outputs.model_output}}",
    environment="aml-scikit-learnv2@latest",
    experiment_name="train_model_credit_default_prediction",
    display_name="credit_default_prediction",
)


# In[19]:


# ml_client.create_or_update(job)


# In[6]:


from azure.ai.ml import command, Input, dsl
from azure.ai.ml.entities import PipelineJob


@dsl.pipeline(
    compute="cpu-cluster1",  # replace with your cluster name
    description="Pipeline for credit default prediction",
)
def credit_default_pipeline():
    train_job = train_step()
    # later you can add more steps like:
    # evaluate_job = evaluate_component(inputs=...)
    # evaluate_job.run_after(train_job)
    return {"model_output": train_job.outputs.model_output}

# Create pipeline job instance
pipeline_job: PipelineJob = credit_default_pipeline()

# Submit pipeline job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline job submitted: {submitted_job.name}")


# In[7]:


import uuid

# Creating a unique name for the endpoint
online_endpoint_name = "credit-endpoint-" + str(uuid.uuid4())[:8]


# In[8]:


from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is an online endpoint",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
        "model_type": "sklearn.GradientBoostingClassifier",
    },
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")


# In[9]:


endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)


# In[10]:


# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)

print(latest_model_version)


# In[17]:


# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)


# create an online deployment.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    environment="aml-scikit-learnv2@latest",
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()

