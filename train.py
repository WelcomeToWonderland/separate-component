import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datetime import datetime
from alive_progress import alive_bar
from model.han import make_model  # 导入自定义模型

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, parent_dir, transform=None):
        self.image_dir = os.path.join(parent_dir, 'img')
        self.label_dir = os.path.join(parent_dir, 'label')
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 将图像转换为Tensor
        ])
        self.image_list = [f for f in os.listdir(self.label_dir) if f.endswith('.png')]  # 从label文件夹中获取文件名列表

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_basename = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_basename)
        label_path = os.path.join(self.label_dir, image_basename)
        
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

# 定义训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=100, device='cuda'):
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        model.train()
        
        running_loss = 0.0
        with alive_bar(len(dataloader), title="Training") as bar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                bar()

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        
        # 保存当前模型权重
        torch.save(model.state_dict(), os.path.join(log_dir, f'model_epoch_{epoch}.pth'))
        
        # 保存最佳模型权重
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(log_dir, 'best_model.pth'))
    
    # 训练结束后加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 定义推理函数
def infer_model(model, dataloader, device='cuda'):
    model.eval()
    inference_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'), 'inference')
    os.makedirs(inference_dir, exist_ok=True)

    with torch.no_grad():
        with alive_bar(len(dataloader), title="Inference") as bar:
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # 保存推理结果
                output_image = transforms.ToPILImage()(outputs.squeeze(0).cpu())
                output_image.save(os.path.join(inference_dir, f'infer_{i}.png'))
                bar()

# 主函数
def main():
    parent_dir = '/path/to/your/dataset'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集和数据加载器
    train_dataset = ImageDataset(parent_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 测试集数据加载器，batch size设置为1
    test_dataset = ImageDataset(parent_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 使用make_model函数生成模型
    model = make_model()
    
    # 使用所有可用的GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model = train_model(model, train_loader, criterion, optimizer, num_epochs=100, device=device)
    
    # 加载最佳权重并对测试集进行推理
    infer_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()
