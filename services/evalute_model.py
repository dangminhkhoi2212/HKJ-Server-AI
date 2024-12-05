import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.image_serach_service import ImageSearchService


class ImageSearchEvaluationService:
    def __init__(self, index_file):
        self.image_search_service = ImageSearchService()
        self.image_extract_service = ImageExtractService()
        self.supabase = SupabaseClient()
        self.index = index_file

    def open_image(self, image_path: str):
        """Convert image at image_path to bytes."""
        try:
            with Image.open(image_path) as img:
                img_file_like = io.BytesIO()
                img.save(img_file_like, img.format)
                img_file_like.seek(0)
                return img_file_like
        except Exception as e:
            raise ValueError(f"Error converting image to bytes: {e}")

    def extract_id_from_filename(self, filename: str) -> str:
        """Extract ID from filename format (id).jpg."""
        try:
            print(f'extract_id_from_filename: {filename}')
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

    def find_jewelry_same_category(self, ids) -> List[str]:
        response = self.supabase.client.table("hkj_jewelry_model").select("*").in_('id', ids).execute()
        return [str(item['category_id']) for item in response.data]

    def search_image(self, file, k=10) -> List[str]:
        """Search image in Supabase storage and return results."""
        try:
            # Convert to image without background
            file_bytes = file.read()
            image_no_bg = self.image_extract_service.remove_bg_image(file_bytes)

            # Open the original image
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

            if image is None and image_no_bg is None:
                raise ValueError('Image not found')

            # Use ThreadPoolExecutor to perform searches in parallel
            with ThreadPoolExecutor() as executor:
                future_no_bg = executor.submit(self.image_search_service.search_with_faiss, image_no_bg, k)
                future_with_bg = executor.submit(self.image_search_service.search_with_faiss, image, k)

                # Collect results as they complete
                search_no_bg = future_no_bg.result()
                search_with_bg = future_with_bg.result()

            # Combine and sort results
            result = search_with_bg + search_no_bg
            result = sorted(result, key=self.image_search_service.sorted_field)
            print(f'result: {result}')

            # Unique ids
            # Unique ids while maintaining order
            # seen = set()
            # unique_ids = []
            # for item in result:
            #     if item['id'] not in seen:
            #         seen.add(item['id'])
            #         unique_ids.append(item['id'])
            #
            # print(f'Unique ids: {unique_ids}')
            # Fetch jewelry models from Supabase
            ids = [item['id'] for item in result]
            category_ids = self.find_jewelry_same_category(ids)
            print(f'category_ids: {category_ids}')
            return category_ids
        except Exception as e:
            raise Exception(f"Error searching image: {str(e)}")

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
                img_file_like = self.open_image(image_path)
                search_results = self.search_image(img_file_like, max_k)

                query_metrics = {
                    'query_id': actual_id,
                    'filename': filename,
                    'precision_at_k': {}
                }

                for k in k_values:
                    relevant_items = search_results[:k].count(actual_id)
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

    def evaluate_model_2(self, test_folder: str, k_values: List[int] = [1, 5, 10]) -> Dict:
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
                print(f'actual_id: {actual_id}')
                image_path = os.path.join(test_folder, filename)
                img_file_like = self.open_image(image_path)
                search_results = self.search_image(img_file_like, max_k)
                relevant_items_in_top_k_recall = search_results.count(actual_id)
                # Total number of relevant items (assume only 1 relevant item per query)
                total_relevant_items = 1

                for k in k_values:
                    # Relevant items in Top K
                    relevant_items_in_top_k = search_results[:k].count(actual_id)
                    print(f'''relevant_items_in_top_k: {relevant_items_in_top_k}''')

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

    def evaluate_final(self, test_folder):
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder does not exist: {test_folder}")

        all_true_labels = []
        all_predicted_labels = []

        for filename in os.listdir(test_folder):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            try:
                true_label = int(self.extract_id_from_filename(filename))
                image_path = os.path.join(test_folder, filename)
                img_file_like = self.open_image(image_path)
                result = self.image_search_service.search_image(img_file_like, 10)
                predicted_labels = [x['category_id'] for x in result]
                print(f'predicted_labels: {predicted_labels}')

                all_true_labels.extend([true_label] * len(predicted_labels))
                all_predicted_labels.extend(predicted_labels)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Tính toán các chỉ số bằng thư viện sklearn
        precision = precision_score(all_true_labels, all_predicted_labels, average='macro')
        recall = recall_score(all_true_labels, all_predicted_labels, average='macro')
        f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }


def main():
    search_service = ImageSearchService()
    index = search_service.index
    evaluator = ImageSearchEvaluationService(index)
    test_folder = r"E:/Images/test"

    metrics = evaluator.evaluate_final(test_folder)
    # evaluator.print_evaluation_results_2(metrics)


if __name__ == "__main__":
    main()
