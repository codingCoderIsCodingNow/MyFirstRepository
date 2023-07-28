import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import  numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
def generate_dataset(num_samples, curve_func, noise_std):
    x = np.linspace(-10, 10, num_samples)
    y = curve_func(x) + np.random.normal(0, noise_std, num_samples)
    dataset = np.column_stack((x, y))
    return dataset

def curve_func(x):
    return x**2 + 2*x - 5  # 曲线函数的定义

# 设置参数
num_samples = 100000 # 数据点数量
noise_std = 1  # 噪声标准差
latent_dim = 1
input_dim = 2
# 生成数据集
dataset1 = generate_dataset(num_samples, curve_func, noise_std)
data = torch.tensor(dataset1, dtype=torch.float32).to(device)

# 加载预训练的 Generator 模型
generator_model = 'generator_model.pth'
generator = Generator(input_dim= latent_dim, output_dim= input_dim)
generator.load_state_dict(torch.load(generator_model))
generator.to(device)
generator.eval()

with torch.no_grad():
    noise = torch.randn(1000, latent_dim).to(device)
    generated_samples = generator(noise).cpu()

input_min = -200
input_max = 200
output_min = -200
output_max = 200

# 可视化输入、输出的对比
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), s=5)
axes[0].set_title("Input Data")
axes[0].set_xlim(input_min, input_max)
axes[0].set_ylim(output_min, output_max)

axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=5)
axes[1].set_title("Generated Samples")
axes[1].set_xlim(input_min, input_max)
axes[1].set_ylim(output_min, output_max)

plt.show()

def calculate_r_squared(generated_samples, target):
    y_true = target.detach().numpy()
    y_pred = generated_samples.detach().numpy()
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# 将数据移动到 CPU
target = data.cpu()[:1000]

# 计算拟合优度
r_squared = calculate_r_squared(generated_samples, target)
print("R-squared: {:.4f}".format(r_squared))

# 计算 MSE Loss
recon_loss = nn.MSELoss()(generated_samples, target)
print("Reconstruction Loss: {:.4f}".format(recon_loss.item()))
