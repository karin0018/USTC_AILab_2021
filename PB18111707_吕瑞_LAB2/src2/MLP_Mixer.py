
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
#禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mixer_Layer(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim,drop_rate=0.):
        """
        核心是将 token-mixing 的信息和 channel-mixing 的信息分开提取
        token 指一系列线性映射后的 patches
        channel-mixing 是为了让 mlp 结构可以在通道之间学习特征
        token-mixing 可以满足 mlp 对不同空域位置的学习
        """
        super(Mixer_Layer, self).__init__()
        ########################################################################
        #这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.ln_token=nn.LayerNorm(hidden_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.token_mix=nn.Sequential(
            nn.Linear(num_tokens,tokens_mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(tokens_mlp_dim,num_tokens),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(channels_mlp_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        ########################################################################

    def forward(self, x):
        ########################################################################
        out = self.ln_token(x).transpose(1,2)
        x = x+self.token_mix(out).transpose(1,2)
        out2 = self.ln_channel(x)
        x = x + self.channel_mix(out2)
        return x
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, depth, num_class,hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=28,drop_rate=0.):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        #这里写Pre-patch Fully-connected, Global average pooling, fully connected
        num_tokens = (image_size // patch_size)**2 # 序列数

        self.patch_emb=nn.Conv2d(1,hidden_dim,kernel_size=patch_size,stride=patch_size,bias=False)
        self.mlp = nn.Sequential(*[Mixer_Layer(num_tokens,hidden_dim,tokens_mlp_dim,channels_mlp_dim,drop_rate) for _ in range(depth)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim,num_class)

        ########################################################################


    def forward(self, data):
        ########################################################################
        #注意维度的变化
        data = self.patch_emb(data)
        data = data.flatten(2).transpose(1,2)
        data = self.mlp(data)
        data = self.ln(data)
        data = data.mean(dim=1)
        data = self.fc(data)
        return data

        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            #计算loss并进行优化
            # 梯度置为 0
            optimizer.zero_grad()
            pre_res = model(data)
            # 计算交叉熵 loss - criterion
            loss = criterion(pre_res,target) # pre_res:预测结果；target：标签
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0 #correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
        ########################################################################
        #需要计算测试集的loss和accuracy
            pre_res = model(data)
            test_loss += criterion(pre_res, target)  # pre_res:预测结果；target：标签
            predicted = torch.max(pre_res.data, 1)[1] # 选出每行最大值作为预测结果
            num_correct += (predicted == target).sum().item()

        # 计算平均 loss 和 准确率
        test_loss = test_loss/len(test_loader)
        accuracy = num_correct/len(test_loader.dataset)

        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))




if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root = './data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = MNIST(root = './data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    ########################################################################
    model = MLPMixer(num_class=10, patch_size=7, hidden_dim=8, tokens_mlp_dim=32,\
        channels_mlp_dim=32, depth=8, image_size=28, drop_rate=0.).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
