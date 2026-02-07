
try:
    from databricks.sdk import WorkspaceClient
    print("Databricks SDK is installed.")
except ImportError:
    print("Databricks SDK is NOT installed.")
