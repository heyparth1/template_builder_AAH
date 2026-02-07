
import os
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import DatabricksError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabricksClient:
    """
    Client for interacting with Databricks Workspace features, specifically Volumes.
    """
    def __init__(self, host: str = None, token: str = None):
        """
        Initialize Databricks Workspace Client.
        defaults to environment variables DATABRICKS_HOST and DATABRICKS_TOKEN if not provided.
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        
        if not self.host or not self.token:
            logger.warning("Databricks credentials missing. Client will not work.")
            self.client = None
        else:
            try:
                self.client = WorkspaceClient(
                    host=self.host,
                    token=self.token
                )
                logger.info("Databricks Workspace Client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Databricks Client: {e}")
                self.client = None

    def upload_file_to_volume(self, local_path: str, volume_path: str, overwrite: bool = True) -> str:
        """
        Uploads a local file to a Databricks Volume.
        
        Args:
            local_path: Path to the local file to upload.
            volume_path: Target path in Databricks Volume (e.g., /Volumes/catalog/schema/volume/file.ext).
            overwrite: Whether to overwrite existing file.
            
        Returns:
            The volume path of the uploaded file if successful.
            
        Raises:
            Exception if upload fails.
        """
        if not self.client:
            raise Exception("Databricks Client not initialized due to missing credentials.")
            
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
            
        try:
            # Ensure volume path starts with /Volumes/
            if not volume_path.startswith("/Volumes/"):
                logger.warning(f"Volume path '{volume_path}' does not start with /Volumes/. This might be incorrect.")

            logger.info(f"Uploading {local_path} to {volume_path}...")
            
            with open(local_path, "rb") as f:
                self.client.files.upload(
                    volume_path,
                    f,
                    overwrite=overwrite
                )
                
            logger.info(f"Successfully uploaded to {volume_path}")
            return volume_path
            
        except DatabricksError as e:
            logger.error(f"Databricks API Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

# Singleton instance
_client_instance = None

def get_databricks_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = DatabricksClient()
    return _client_instance

def upload_to_volume(local_path: str, volume_path: str) -> str:
    """
    Convenience function to upload file to volume using singleton client.
    """
    client = get_databricks_client()
    return client.upload_file_to_volume(local_path, volume_path)
