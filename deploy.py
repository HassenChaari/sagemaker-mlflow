import argparse
import mlflow.sagemaker
from sagemaker.model_monitor import DataCaptureConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="region where to deploy the endpoint", required=True)
    parser.add_argument("--aws-id", help="Account ID of the AWS Account", required=True)
    parser.add_argument("--role", help="ARN ROLE for sagemaker", required=True)
    parser.add_argument("--app-name", help="Specify the application name", required=True)
    parser.add_argument("--uri", help="model uri", required=True)
    parser.add_argument("--image-name", help="Image Name for pyfunc-mlflow", required=True)
    parser.add_argument("--tag", help="Image Tag for pyfuncf-mlflow", required=True)
    parser.add_argument("--capture-path", help="Capture data location")
    return parser.parse_args()


def data_capture_config(path):
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=path
    )
    return data_capture_config._to_request_dict()


if __name__ == "__main__":
    args = get_args()

    data_capture_config_dict = data_capture_config(args.capture_path)
    image_uri = f"{args.aws_id}.dkr.ecr.{args.region}.amazonaws.com/{args.image_name}:{args.tag}"
    mlflow.sagemaker.deploy(
        app_name=args.app_name,
        model_uri=args.uri,
        region_name=args.region,
        mode="create",
        execution_role_arn=args.role,
        instance_type="ml.t2.medium",
        data_capture_config=data_capture_config_dict,
        image_url=image_uri
    )
