import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义Generator模型
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

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# 自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 设置训练参数
input_dim = 2
latent_dim = 1
batch_size = 128
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成自定义数据集
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

# 生成数据集
dataset1 = generate_dataset(num_samples, curve_func, noise_std)
data = torch.tensor(dataset1, dtype=torch.float32).to(device)
dataset = CustomDataset(data)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化Generator和Discriminator模型
generator = Generator(latent_dim, input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

# 训练模型
for epoch in range(epochs):
    for batch_idx, real_data in enumerate(train_loader):
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # 训练Discriminator
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 计算真实数据的损失
        real_output = discriminator(real_data)
        real_loss = adversarial_loss(real_output, real_labels)

        # 生成噪声并生成假数据
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise)

        # 计算假数据的损失
        fake_output = discriminator(fake_data.detach())
        fake_loss = adversarial_loss(fake_output, fake_labels)

        # 更新Discriminator的参数
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # 训练Generator
        generator_optimizer.zero_grad()
        fake_output = discriminator(fake_data)
        generator_loss = adversarial_loss(fake_output, real_labels)

        # 更新Generator的参数
        generator_loss.backward()
        generator_optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
                epoch+1, epochs, batch_idx+1, len(train_loader), generator_loss.item(), discriminator_loss.item()
            ))

# 保存Generator模型
torch.save(generator.state_dict(), 'generator_model.pth')
print("保存成功")
# 使用Generator生成样本
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

# 计算拟合优度
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

