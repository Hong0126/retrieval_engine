"""This script takes the trained model and deploys it to AI Platform for serving."""

import argparse

from google.cloud import aiplatform


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--gcp-project-id",
        type=str,
        help="The ID of the Google Cloud project.",
        required=True,
    )
    arg_parser.add_argument(
        "--location",
        type=str,
        help="The location of the AI Platform model.",
        default="europe-west4",
    )

    return arg_parser.parse_args()


def main(gcp_project_id: str, location: str) -> None:
    """Main entry point of the script.

    Args:
        gcp_project_id (str): The ID of the Google Cloud project.
        location (str): The location of the AI Platform model.
    """
    aiplatform.init(project=gcp_project_id, location=location)

    model = aiplatform.Model.upload(
        display_name="blogpost-challenge",
        serving_container_image_uri=f"{location}-docker.pkg.dev/{gcp_project_id}/blogpost-search/retrieval-engine:latest",
    )

    model.deploy(
        machine_type="n1-standard-4",
    )


if __name__ == "__main__":
    arguments = vars(parse_arguments())
    main(**arguments)
