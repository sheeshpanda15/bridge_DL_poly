import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# 1. 定义网络结构（要和原来训练时完全一样）
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()  # 切换为评估模式

# 加载图片

image = Image.open("jun7.jpg").convert("L")
image = ImageOps.invert(image)  # 反转颜色（如果是白底黑字）

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_tensor = transform(image).unsqueeze(0).to(device)



# 使用模型预测
with torch.no_grad():
    output = model(image_tensor)
    predicted = torch.argmax(output, 1)
    print(f"预测结果是：{predicted.item()}")