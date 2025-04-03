from PIL import Image
import torch
from torchvision import transforms
from lung_classifier.singletons.lung_models import CT_CNN
import joblib
import pandas as pd
import requests
import io
import os

class LungCancerPredictor:
    def __init__(self, data : dict, path : str) -> None:
        self.data = data
        self.path = os.path.join(path, "models")

        # Mapping for categorical values
        self.gender_map = {"M": 1, "F": 0}
        print(os.curdir)

        # Transform the data to match the required feature names
        self.formatted_data = {
            "GENDER": [self.gender_map[self.data["gender"]]],
            "AGE": [self.data["age"]],
            "SMOKING": [self.data["smoking"]],
            "YELLOW_FINGERS": [self.data["yellow_fingers"]],
            "ANXIETY": [self.data["anxiety"]],
            "PEER_PRESSURE": [self.data["peer_pressure"]],
            "CHRONIC_DISEASE": [self.data["chronic_disease"]],
            "FATIGUE": [self.data["fatigue"]],
            "ALLERGY": [self.data["allergy"]],
            "WHEEZING": [self.data["wheezing"]],
            "ALCOHOL_CONSUMING": [self.data["alcohol_consuming"]],
            "COUGHING": [self.data["coughing"]],
            "SHORTNESS_OF_BREATH": [self.data["shortness_of_breath"]],
            "SWALLOWING_DIFFICULTY": [self.data["swallowing_difficulty"]],
            "CHEST_PAIN": [self.data["chest_pain"]]
        }

        root_dir_relative = "D:/mission_2025/python_learning/django_services/lung_cancer_detector/models/"

        self.rf = joblib.load(os.path.join(self.path, "random_forest.pkl"))
        self.gb = joblib.load(os.path.join(self.path, "gradient_boosting.pkl"))
        self.ada = joblib.load(os.path.join(self.path, "adaboost.pkl"))

        self.sample_data = pd.DataFrame(self.formatted_data)

        # Normalize input using the saved scaler
        self.scaler = joblib.load(root_dir_relative + "scaler.pkl")
        self.sample_scaled = self.scaler.transform(self.sample_data)

        print("Data has been collected and scaled..")

        self.model = torch.load(root_dir_relative + "lung_cancer_detector.pth")

        # Convert to PyTorch tensor
        self.sample_tensor = torch.tensor(self.sample_scaled, dtype=torch.float32)

        print("Models are loaded perfectly")

    def predict(self) -> str:
        # Predict using Neural Network
        """
        with torch.no_grad():
            nn_prediction = self.model(self.sample_tensor).item()
            nn_prediction = 1 if nn_prediction > 0.5 else 0  # Convert probability to class
        """

        # Predict using ML Models
        rf_pred = self.rf.predict(self.sample_scaled)[0]
        gb_pred = self.gb.predict(self.sample_scaled)[0]
        ada_pred = self.ada.predict(self.sample_scaled)[0]

        # Ensemble (average voting)
        ensemble_pred = round((rf_pred + gb_pred + ada_pred) / 3)
        return 'YES' if ensemble_pred == 1 else 'NO'

class LungCancerCTPredictor:
    def __init__(self, image : str, path : str) -> None:
        self.image = image
        self.path = os.path.join(path, "models", "lung_cancer_detector.pth")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ",device)

        self.model = CT_CNN(num_classes=3).to(device)
        print("Models loaded")

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

         # Define class labels
        self.class_labels = ["Benign", "Malignant", "Normal"]
        self.output_dir = "D:/mission_2025/python_learning/django_services/lung_cancer_detector/models/lung_cancer_detector.pth"
        self.model.load_state_dict(torch.load(self.path))
        print("Model is fitted with weights")

    # Load and transform the image
    def preprocess_image(self):
        response = requests.get(self.image)
        byteInfo = io.BytesIO(response.content)
        image = Image.open(byteInfo).convert("RGB")  # Ensure RGB format
        image = self.transform(image)  # Apply transformations
        image = image.unsqueeze(0)  # Add batch dimension
        print("Image has been prepared")
        return image

    def predict_image(self) -> str:
        image = self.preprocess_image()

        # Forward pass through the model
        with torch.no_grad():
            print("Output is to be found")
            output = self.model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        return self.class_labels[predicted_class]