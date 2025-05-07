import os

# External Dependencies:
import boto3
from botocore.config import Config

from dotenv import load_dotenv
load_dotenv()


def get_bedrock_client_from_environ():
    """
    Create a boto3 client for Amazon Bedrock, using the access key & secret from envvars.
    Required envvars:
    - `BEDROCK_IAM_ACCESS_KEY`
    - `BEDROCK_IAM_SECRET_KEY`
    """
    session = boto3.Session(
        aws_access_key_id=os.environ.get("BEDROCK_IAM_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("BEDROCK_IAM_SECRET_KEY"),
    )
    boto3_bedrock = session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )

    return boto3_bedrock
