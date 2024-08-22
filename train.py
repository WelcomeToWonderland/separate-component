import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from datetime import datetime
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from model.han import make_model  # 导入自定义模型

# 获取当前文件所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, parent_dir, output_size=(128, 128), transform=None):
        self.image_dir = os.path.join(parent_dir, 'img')
        self.label_dir = os.path.join(parent_dir, 'label')
        self.output_size = output_size
        self.transform = transform or transforms.Compose([
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

        # 使用预处理方法处理图像和标签
        image, label = self.preprocess_image(image, label)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
    
    def preprocess_image(self, image, label):
        # 将所有非白色像素转换为黑色像素
        image_np = np.array(image)
        image_np[(image_np != [255, 255, 255]).any(axis=-1)] = [0, 0, 0]
        image = Image.fromarray(image_np)
        
        # 灰度化和二值化
        gray_image = image.convert("L")
        _, binary_image_np = cv2.threshold(np.array(gray_image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = Image.fromarray(binary_image_np)
        
        gray_label = label.convert("L")
        _, binary_label_np = cv2.threshold(np.array(gray_label), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_label = Image.fromarray(binary_label_np)

        # 进行随机裁剪
        image, label = self.random_crop(binary_image, binary_label, self.output_size)
        
        return image, label

    def random_crop(self, image, label, output_size):
        width, height = image.size
        
        # 计算填充量以确保输出尺寸大于原始尺寸
        if width < output_size[0]:
            padding_left = max((output_size[0] - width) // 2, 0)
            padding_right = output_size[0] - width - padding_left
        else:
            padding_left = padding_right = 0

        if height < output_size[1]:
            padding_top = max((output_size[1] - height) // 2, 0)
            padding_bottom = output_size[1] - height - padding_top
        else:
            padding_top = padding_bottom = 0

        # 对image和label进行同步填充
        image = ImageOps.expand(image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=255)
        label = ImageOps.expand(label, border=(padding_left, padding_top, padding_right, padding_bottom), fill=255)

        # 获取填充后的尺寸
        width, height = image.size
        
        # 进行随机裁剪
        left = np.random.randint(0, width - output_size[0] + 1)
        top = np.random.randint(0, height - output_size[1] + 1)
        right = left + output_size[0]
        bottom = top + output_size[1]
        
        image = image.crop((left, top, right, bottom))
        label = label.crop((left, top, right, bottom))
        
        return image, label

# 定义训练和测试函数
def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, log_dir, num_epochs=100, device='cuda'):
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        
        running_loss = 0.0
        with alive_bar(len(train_loader), title="Training") as bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                bar()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        print(f'Training Loss: {epoch_train_loss:.4f}')
        
        # 测试模型
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            with alive_bar(len(test_loader), title="Testing") as bar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_test_loss += loss.item() * inputs.size(0)
                    bar()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        print(f'Test Loss: {epoch_test_loss:.4f}')
        
        # 保存当前模型权重
        torch.save(model.state_dict(), os.path.join(log_dir, f'model_epoch_{epoch}.pth'))
        
        # 保存最佳模型权重
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(log_dir, 'best_model.pth'))
    
    # 训练结束后加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 可视化损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_plot.png'))
    plt.show()
    
    return model

# 定义推理函数
def infer_model(model, dataloader, log_dir, device='cuda'):
    model.eval()
    inference_dir = os.path.join(log_dir, 'inference')
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
    parent_dir = os.path.join(current_dir, 'datasets', 'only-component')  # 数据集目录
    parent_dir_train = os.path.join(parent_dir, 'train')
    parent_dir_test = os.path.join(parent_dir, 'test')
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用绝对路径创建日志目录
    log_dir = os.path.join(current_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)

    # 数据集和数据加载器
    train_dataset = ImageDataset(parent_dir_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 测试集数据加载器，batch size设置为1
    test_dataset = ImageDataset(parent_dir_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 使用make_model函数生成模型
    model = make_model(None)
    
    # 使用所有可用的GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练和测试模型
    trained_model = train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, log_dir, num_epochs=100, device=device)
    
    # 执行推理
    infer_model(trained_model, test_loader, log_dir, device=device)

if __name__ == "__main__":
    main()
