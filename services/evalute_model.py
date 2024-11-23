import io
import os
from typing import List, Dict

import numpy as np
from PIL import Image

from constants import name
from helpers.file_helper import load_file
from services.image_extract_service import ImageExtractService
from services.image_serach_service2 import ImageSearchService


class ImageSearchEvaluationService:
    def __init__(self, index_file):
        self.image_search_service = ImageSearchService()
        self.image_extract_service = ImageExtractService()
        self.index = index_file

    def image_to_bytes(self, image_path: str) -> bytes:
        """Convert image at image_path to bytes."""
        try:
            with Image.open(image_path) as image:
                byte_stream = io.BytesIO()
                image.save(byte_stream, format=image.format)
                return byte_stream.getvalue()
        except Exception as e:
            raise ValueError(f"Error converting image to bytes: {e}")

    def extract_id_from_filename(self, filename: str) -> str:
        """Extract ID from filename format (id).jpg."""
        try:
            return str(filename.split('.')[0])
        except ValueError as e:
            raise ValueError(f"Invalid filename format: {e}")

    def calculate_precision_at_k(self, relevant_items: int, k: int) -> float:
        """Calculate Precision@K metric."""
        return relevant_items / k if k > 0 else 0.0

    def calculate_average_precision(self, actual_id: int, retrieved_results: List[Dict], max_k: int) -> float:
        """Calculate Average Precision for a single query."""
        relevant_count = 0
        sum_precision = 0.0

        for k, result in enumerate(retrieved_results[:max_k], 1):
            if result == actual_id:
                relevant_count += 1
                sum_precision += self.calculate_precision_at_k(relevant_count, k)

        return sum_precision / min(len(retrieved_results), max_k)

    def search_image(self, file: bytes, top_k: int) -> List[str]:
        """Search image and return top_k results."""
        try:
            search_vector = np.atleast_2d(self.image_extract_service.extract_vector(file))
            index = self.index

            index.nprobe = 1
            distances, indices = index.search(search_vector, top_k)

            return [
                str(idx)
                for idx, dist in zip(indices[0], distances[0])
            ]
        except Exception as e:
            raise RuntimeError(f"Error searching image: {e}")

    def evaluate_model(self, test_folder: str, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Evaluate the model using P@K and MAP metrics."""
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder does not exist: {test_folder}")

        k_values = sorted(k_values)
        max_k = max(k_values)
        results = {
            'precision_at_k': {k: [] for k in k_values},
            'average_precision': [],
            'details': []
        }

        for filename in os.listdir(test_folder):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            try:
                actual_id = self.extract_id_from_filename(filename)
                image_path = os.path.join(test_folder, filename)
                image_bytes = self.image_to_bytes(image_path)
                image_bytes = self.image_extract_service.remove_bg_image(image_bytes)
                search_results = self.search_image(image_bytes, max_k)

                query_metrics = {
                    'query_id': actual_id,
                    'filename': filename,
                    'precision_at_k': {}
                }

                for k in k_values:
                    relevant_items = sum(1 for pred_id in search_results[:k] if pred_id == actual_id)
                    precision = self.calculate_precision_at_k(relevant_items, k)
                    results['precision_at_k'][k].append(precision)
                    query_metrics['precision_at_k'][k] = precision

                ap = self.calculate_average_precision(actual_id, search_results, max_k)
                results['average_precision'].append(ap)
                query_metrics['average_precision'] = ap

                results['details'].append(query_metrics)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return self._calculate_final_metrics(results)

    def evaluate_model_2(self, test_folder: str, k_values: List[int] = [5, 10, 15, 30]) -> Dict:
        """Evaluate the model and return Precision@K, Recall@K, and F1@K."""
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder does not exist: {test_folder}")

        k_values = sorted(k_values)
        max_k = max(k_values)
        metrics = {
            'precision_at_k': {k: [] for k in k_values},
            'recall_at_k': {k: [] for k in k_values},
            'f1_at_k': {k: [] for k in k_values},
        }

        # Assuming 1 relevant item (ground truth) per query
        for filename in os.listdir(test_folder):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            try:
                actual_id = self.extract_id_from_filename(filename)
                image_path = os.path.join(test_folder, filename)
                image_bytes = self.image_to_bytes(image_path)
                image_bytes = self.image_extract_service.remove_bg_image(image_bytes)
                search_results = self.search_image(image_bytes, max_k)
                relevant_items_in_top_k_recall = sum(1 for pred_id in search_results if pred_id == actual_id)
                # Total number of relevant items (assume only 1 relevant item per query)
                total_relevant_items = 1

                for k in k_values:
                    # Relevant items in Top K
                    relevant_items_in_top_k = sum(1 for pred_id in search_results[:k] if pred_id == actual_id)

                    # Precision@K
                    precision = self.calculate_precision_at_k(relevant_items_in_top_k, k)

                    # Recall@K
                    recall = (
                        relevant_items_in_top_k / relevant_items_in_top_k_recall
                        if relevant_items_in_top_k_recall > 0 else 0.0
                    )

                    # F1@K
                    f1 = (
                        (2 * precision * recall) / (precision + recall)
                        if (precision + recall) > 0 else 0.0
                    )

                    metrics['precision_at_k'][k].append(precision)
                    metrics['recall_at_k'][k].append(recall)
                    metrics['f1_at_k'][k].append(f1)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        mean_metrics = {
            'mean_precision_at_k': {
                k: np.mean(values) if values else 0.0
                for k, values in metrics['precision_at_k'].items()
            },
            'mean_recall_at_k': {
                k: np.mean(values) if values else 0.0
                for k, values in metrics['recall_at_k'].items()
            },
            'mean_f1_at_k': {
                k: np.mean(values) if values else 0.0
                for k, values in metrics['f1_at_k'].items()
            },
        }

        return mean_metrics

    def _calculate_final_metrics(self, results: Dict) -> Dict:
        """Calculate final metrics: MAP and mean P@K."""
        return {
            'map': np.mean(results['average_precision']) if results['average_precision'] else 0.0,
            'mean_precision_at_k': {
                k: np.mean(values) if values else 0.0
                for k, values in results['precision_at_k'].items()
            },
            'query_details': results['details']
        }

    def print_evaluation_results(self, metrics: Dict):
        """Print formatted evaluation results."""
        print("\n=== Image Search Model Evaluation Results ===\n")
        print(f"Mean Average Precision (MAP): {metrics['map']:.4f}")
        print("\nMean Precision@K:")
        for k, value in metrics['mean_precision_at_k'].items():
            print(f"P@{k}: {value:.4f}")
        print("\nDetailed Results by Query:")
        for query in metrics['query_details']:
            print(f"\nQuery {query['filename']}:")
            print(f"  Average Precision: {query['average_precision']:.4f}")
            for k, p in query['precision_at_k'].items():
                print(f"  P@{k}: {p:.4f}")

    def print_evaluation_results_2(self, metrics: Dict):
        """Print formatted Precision@K, Recall@K, and F1@K results."""
        print("\n=== Evaluation Results ===\n")

        print("Mean Precision@K:")
        for k, value in metrics['mean_precision_at_k'].items():
            print(f"P@{k}: {value:.4f}")

        print("\nMean Recall@K:")
        for k, value in metrics['mean_recall_at_k'].items():
            print(f"R@{k}: {value:.4f}")

        print("\nMean F1@K:")
        for k, value in metrics['mean_f1_at_k'].items():
            print(f"F1@{k}: {value:.4f}")


def main():
    index = load_file(name.FILE_FAISS_INDEX)
    evaluator = ImageSearchEvaluationService(index)
    test_folder = r"E:/Images/test"

    metrics = evaluator.evaluate_model_2(test_folder)
    evaluator.print_evaluation_results_2(metrics)


if __name__ == "__main__":
    main()
