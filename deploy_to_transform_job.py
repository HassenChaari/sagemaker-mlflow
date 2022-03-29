import argparse
import mlflow.sagemaker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job-name", help="Specify the application name", required=True)
    parser.add_argument("uri", help="model uri", required=True)
    parser.add_argument("data-type", help="input data type", required=True)
    parser.add_argument("input-uri", help="input uri", required=True)
    parser.add_argument("content-type", help="content type", required=True)
    parser.add_argument("output-uri", help="output uri", required=True)
    parser.add_argument("region", help="region where to deploy the endpoint", required=True)
    parser.add_argument("aws-id", help="Account ID of the AWS Account", required=True)
    parser.add_argument("role", help="ARN ROLE for sagemaker", required=True)
    parser.add_argument("image-name", help="Image Name for pyfunc-mlflow", required=True)
    parser.add_argument("tag", help="Image Tag for pyfuncf-mlflow", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    image_uri = f"{args.aws_id}.dkr.ecr.{args.region}.amazonaws.com/{args.image_name}:{args.tag}"
    mlflow.sagemaker.deploy_transform_job(
        job_name=args.job_name,
        model_uri=args.uri,
        s3_input_data_type=args.data_type,
        s3_input_uri=args.input_uri,
        content_type=args.content_type,
        s3_output_path=args.output_uri,
        region_name=args.region,
        execution_role_arn=args.role,
        instance_type="ml.t2.medium",
        image_url=image_uri
    )
