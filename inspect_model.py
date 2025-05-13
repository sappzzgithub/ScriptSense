# inspect_model.py
import torch

model_path = "/Users/sakshizanjad/Desktop/project_root/models/personality_cnn.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

print("🔍 Model Keys in state_dict:\n")
for key in state_dict.keys():
    print(key)