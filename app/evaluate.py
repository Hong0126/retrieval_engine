"""Module used to evaluate the performance of the ML6 blog post retrieval engine."""

import base64
import json
import os
import numpy as np
import requests
import time
from typing import List, Dict, Any, Optional

# --- Configuration ---
APP_URL = "http://localhost:63564/predict"  # Adjust if your app runs on a different port
EVALUATION_DATA_PATH = "data/eval_data/evaluation_data.json"
EVALUATION_IMAGES_DIR = "data/eval_data/eval_images"
K_FOR_MRR = 3  # Number of top results to consider for MRR@K


# --- Helper Functions ---
def load_evaluation_data(filepath: str) -> List[Dict[str, Any]]:
    """Loads evaluation data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


def run_test(test_case: Dict[str, Any]) -> Optional[float]:
    """Runs a single test case against the local prediction endpoint."""
    query_id = test_case.get("query_id")
    query_type = test_case.get("query_type")
    query_content = test_case.get("query_content")
    relevant_doc_title = test_case.get("relevant_doc_title").replace(
        ".json", ""
    )

    if not all([query_id, query_type, query_content, relevant_doc_title]):
        print(f"Warning: Incomplete test case - skipping: {test_case}")
        return None  # Indicate skip

    payload = {"instances": [{}]}

    if query_type == "image":
        image_path = os.path.join(EVALUATION_IMAGES_DIR, query_content)
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            payload["instances"][0]["image_bytes"] = {"b64": base64_image}
        else:
            return None  # Indicate failure due to image loading
    elif query_type == "text":
        payload["instances"][0]["text_input"] = query_content
    else:
        print(f"Error: Unknown query type '{query_type}' for test {query_id}")
        return None  # Indicate failure

    try:
        response = requests.post(APP_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        prediction_response = response.json()

        if (
            "predictions" in prediction_response
            and prediction_response["predictions"]
        ):
            ranked_documents = [
                doc.replace(".json", "")
                for doc in prediction_response["predictions"][0].get(
                    "ranked_documents", []
                )
            ]
            if relevant_doc_title in ranked_documents[:K_FOR_MRR]:
                rank = ranked_documents.index(relevant_doc_title) + 1
                rr = 1.0 / rank
                print(f"Test '{query_id}': PASS (Rank {rank})")
                return rr
            else:
                print(
                    f"Test '{query_id}': FAIL - Expected '{relevant_doc_title}', Top {K_FOR_MRR} Got '{ranked_documents[:K_FOR_MRR]}'"
                )
                return 0.0
        else:
            print(
                f"Test '{query_id}': FAIL - Invalid prediction response format: {prediction_response}"
            )
            return 0.0

    except requests.exceptions.RequestException as e:
        print(f"Test '{query_id}': ERROR - Request failed: {e}")
        return 0.0
    except json.JSONDecodeError:
        print(f"Test '{query_id}': ERROR - Failed to decode JSON response.")
        return 0.0
    except Exception as e:
        print(f"Test '{query_id}': ERROR - An unexpected error occurred: {e}")
        return 0.0


def calculate_mrr_at_k(reciprocal_ranks: List[float], k: int) -> float:
    """Calculates Mean Reciprocal Rank at K (including 0.0 for misses)."""
    if not reciprocal_ranks:
        return 0.0
    return np.mean(reciprocal_ranks)


# --- Main Function ---
def main() -> None:
    """Loads evaluation data and runs all tests, calculating MRR@K."""
    evaluation_data = load_evaluation_data(EVALUATION_DATA_PATH)
    total_tests = len(evaluation_data)
    reciprocal_ranks = []
    successful_tests = 0

    print("--- Running Local Tests (MRR@{}) ---".format(K_FOR_MRR))

    # Check if the app is running (basic health check)
    try:
        health_response = requests.get(APP_URL.replace('/predict', '/health'))
        health_response.raise_for_status()
        if health_response.json().get("status") == "OK":
            print("App is running and healthy.")
        else:
            print("Warning: App health check failed.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to the app at {APP_URL.replace('/predict', '/health')}. Is it running?")
        return

    for test_case in evaluation_data:
        result = run_test(test_case)
        if result is not None:
            reciprocal_ranks.append(result)
            if result > 0:
                successful_tests += 1
        time.sleep(0.1)  # Be gentle on the local server

    mrr_at_k = calculate_mrr_at_k(reciprocal_ranks, K_FOR_MRR)
    pass_rate = (successful_tests / total_tests) if total_tests > 0 else 1.0

    # You could consider adding more granular evaluation metrics to differentiate between text and image queries
    print("\n--- Evaluation Results ---")
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Pass Rate: {pass_rate:.2%}")
    print(f"MRR@{K_FOR_MRR}: {mrr_at_k:.4f}")


if __name__ == "__main__":
    main()
