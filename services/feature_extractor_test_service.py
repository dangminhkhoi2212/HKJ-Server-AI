import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


class FeatureExtractor:
    def __init__(self):
        self.vgg19_model = self.get_VGG19_model()
        self.resnet50_model = self.get_ResNet50_model()
        self.pca = PCA(n_components=512)

    def get_VGG19_model(self):
        model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        return Model(inputs=model.inputs, outputs=model.output)

    def get_ResNet50_model(self):
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        return Model(inputs=model.inputs, outputs=model.output)

    def preprocess_image(self, img_path, model_type):
        img = Image.open(img_path).resize((224, 224)).convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if model_type == "vgg19":
            x = preprocess_vgg19(x)
        elif model_type == "resnet50":
            x = preprocess_resnet(x)

        return x

    def pca_predict(self, vector):
        reduced_vector = self.pca.fit_transform([vector])[0]  # Giảm chiều vector
        return reduced_vector

    def extract_features(self, img_path):
        # Preprocess image for both models
        img_vgg19 = self.preprocess_image(img_path, "vgg19")
        img_resnet50 = self.preprocess_image(img_path, "resnet50")

        # Extract features using VGG19
        vgg19_features = self.vgg19_model.predict(img_vgg19)[0]
        vgg19_features = vgg19_features / np.linalg.norm(vgg19_features)
        vgg19_features = self.pca_predict(vgg19_features)

        # Extract features using ResNet50
        resnet50_features = self.resnet50_model.predict(img_resnet50)[0]
        resnet50_features = resnet50_features / np.linalg.norm(resnet50_features)
        resnet50_features = self.pca_predict(resnet50_features)
        # Perform PCA
        return vgg19_features, resnet50_features


# Main function to test
def main():
    extractor = FeatureExtractor()
    img_path = "C:/Users/WINDOWS/Desktop/gnxmxmw000205-nhan-vang-trang-10k-dinh-da-ecz-pnj-1.png"  # Replace with the
    # path to your image

    vgg19_features, resnet50_features = extractor.extract_features(img_path)
    print(f"VGG19 Feature Vector (Dim: {len(vgg19_features)})")
    print(f"ResNet50 Feature Vector (Dim: {len(resnet50_features)})")


if __name__ == "__main__":
    main()
