from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from lib.supabase_client import SupabaseClient
from services.jewelry_image_vector_service import JewelryImageVectorService


class UnifiedFeatureExtractor:
    def __init__(self, model_type="resnet50", output_dim=512):
        """
        Khởi tạo feature extractor với kích thước output thống nhất
        
        Parameters:
        - model_type: 'vgg19' hoặc 'resnet50'
        - output_dim: Kích thước vector đặc trưng đầu ra
        """
        self.model_type = model_type.lower()
        self.output_dim = output_dim
        self.input_shape = (224, 224, 3)
        self.model = self._create_model()

    def _create_model(self):
        """Tạo model với lớp projection để thống nhất kích thước đầu ra"""
        if self.model_type == "vgg19":
            base_model = VGG19(weights="imagenet", include_top=False, input_shape=self.input_shape)
            preprocess_fn = vgg19_preprocess
        elif self.model_type == "resnet50":
            base_model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)
            preprocess_fn = resnet_preprocess
        else:
            raise ValueError("model_type phải là 'vgg19' hoặc 'resnet50'")

        # Thêm các lớp để thống nhất kích thước
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # Thêm lớp projection để có kích thước mong muốn
        predictions = Dense(self.output_dim, activation='linear', name='projection')(x)

        # Tạo model mới
        model = Model(inputs=base_model.input, outputs=predictions)

        # Đóng băng các lớp của base model
        for layer in base_model.layers:
            layer.trainable = False

        return model

    def preprocess_input(self, img):
        """Tiền xử lý ảnh tùy theo loại model"""
        if self.model_type == "vgg19":
            return vgg19_preprocess(img)
        return resnet_preprocess(img)

    def image_preprocess(self, img):
        """Chuẩn bị ảnh cho việc trích xuất đặc trưng"""
        img = img.resize((224, 224))
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        return x

    def extract_vector(self, img):
        """
        Trích xuất vector đặc trưng từ ảnh
        
        Parameters:
        - img: PIL Image
        
        Returns:
        - vector: Vector đặc trưng đã chuẩn hóa với kích thước thống nhất
        """
        img_tensor = self.image_preprocess(img)
        vector = self.model.predict(img_tensor)[0]
        # Chuẩn hóa L2
        vector = vector / np.linalg.norm(vector)
        return vector


class ImageExtractService:
    def __init__(self, model_type="resnet50", output_dim=512):
        """
        Khởi tạo service xử lý ảnh
        """
        self.extractor = UnifiedFeatureExtractor(model_type=model_type, output_dim=output_dim)
        self.supabase_client = SupabaseClient()
        self.jewelry_image_vector_service = JewelryImageVectorService()

    def download_and_process_image(self, image_url: str) -> np.ndarray:
        """Download và xử lý ảnh từ URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Trích xuất vectors

            vector = self.extractor.extract_vector(img)

            return np.array(vector)

        except Exception as e:
            raise Exception(f"Lỗi xử lý ảnh: {str(e)}")

    def save_vector_db(self, data: list):
        """Save vector data to Supabase."""
        response = self.jewelry_image_vector_service.save(data)
        return response

    def process_image(self, images: list):
        """Xử lý danh sách ảnh và lưu vectors vào database"""
        for image_data in images:
            print(f'Đang xử lý ảnh: {image_data["url"]}')
            image_url = image_data.get('url')
            image_id = image_data.get('id')
            jewelry_id = image_data.get('jewelryId')

            if self.jewelry_image_vector_service.is_extracted(image_id):
                continue

            vectors = self.download_and_process_image(image_url)

            data = [
                {
                    "image_id": image_id,
                    "parent_id": image_id if index != 0 else None,
                    "vector": vector.tolist(),
                    "jewelry_id": jewelry_id,
                    "has_vector": True,
                    "active": True,

                } for index, vector in enumerate(vectors)
            ]

            self.save_vector_db(data)
            print(f"Đã lưu {len(data)} vectors vào database.")


def main():
    # Test với cả hai model
    vgg_service = ImageExtractService(model_type="vgg19", output_dim=512)
    resnet_service = ImageExtractService(model_type="resnet50", output_dim=512)

    # URL ảnh test
    url = ('https://cdn.pnj.io/images/detailed/197/on-gvctxmc000004-vong-tay-vang-18k-dinh-da-citrine-pnj-audax-rosa-1'
           '.jpg')

    # Trích xuất vectors
    vgg_vectors = vgg_service.download_and_process_image(url)
    resnet_vectors = resnet_service.download_and_process_image(url)

    # In kích thước vectors
    print("VGG19 vector shape:", vgg_vectors.shape)  # Should be (13, 512)
    print("ResNet50 vector shape:", resnet_vectors.shape)  # Should be (13, 512)

    # Tính độ tương đồng
    for i in range(len(vgg_vectors)):
        similarity = np.dot(vgg_vectors[i], resnet_vectors[i])
        print(f"Độ tương đồng giữa VGG19 và ResNet50 vector thứ {i}: {similarity:.4f}")


if __name__ == "__main__":
    main()
