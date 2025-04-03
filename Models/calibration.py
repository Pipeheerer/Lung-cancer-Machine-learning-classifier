import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import your model architecture
from services.lung_classifier.singletons.lung_models import CT_CNN

# Custom dataset for calibration
class CalibrationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            class_path = os.path.join(root_dir, cls)
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(class_path, fname), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create calibration dataset and dataloader
calib_dataset = CalibrationDataset(root_dir="/Users/admin/Downloads/lung_cancer_detector-master/datasets/LungCancer_CT_Dataset/TestDataset", transform=transform)
calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)

# Define a wrapper for temperature scaling
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

def find_temperature(model, valid_loader):
    model.eval()
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        loss = 0
        for images, labels in valid_loader:
            logits = model(images)
            loss += nll_criterion(logits, labels)
        loss = loss / len(valid_loader)
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    print("Optimal temperature:", model.temperature.item())
    return model

# Load your base model (un-calibrated)
base_model = CT_CNN(num_classes=3)
base_model.load_state_dict(torch.load("lung_cancer_detector.pth", map_location=torch.device('cpu')))
base_model.eval()

# Wrap it with temperature scaling
calib_model = ModelWithTemperature(base_model)
calib_model = find_temperature(calib_model, calib_loader)

# Save the calibrated model
torch.save(calib_model.state_dict(), "calibrated_lung_cancer_detector.pth")
print("Calibrated model saved as calibrated_lung_cancer_detector.pth")
