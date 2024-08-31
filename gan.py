# https://qiita.com/miya_ppp/items/f1348e9e73dd25ca6fb5
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import display

batch_size = 64 # バッチサイズ
nz = 100 # 潜在変数の次元数
n_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True # バッチサイズがずれると面倒くさいので最後は切り捨て
)

sample_x, _ = next(iter(dataloader))
w, h = sample_x.shape[2:] # (28, 28)
image_size = w * h        # 784


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), # 1x28x28 -> 784
            nn.Linear(image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(), # 確率なので0~1に
        )

    def forward(self, x):
        y = self.net(x)
        return y

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear(nz, 128),
            self._linear(128, 256),
            self._linear(256, 512),
            nn.Linear(512, image_size),
            nn.Sigmoid() # 濃淡を0~1に
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, 1, w, h) # 784 -> 1x28x28
        return y


# ノイズを生成する関数
def make_noise(batch_size):
    return torch.randn(batch_size, nz, device=device)

# 画像を描画する関数
def write(netG, name, n_rows=1, n_cols=8, size=64):
    z = make_noise(n_rows*n_cols)
    images = netG(z)
    images = transforms.Resize(size)(images)
    img = torchvision.utils.make_grid(images, n_cols)
    img = transforms.functional.to_pil_image(img)
    # display(img)
    img.save(f"{name}.jpg")

real_labels = torch.zeros(batch_size, 1).to(device) # 本物のラベル
fake_labels = torch.ones(batch_size, 1).to(device) # 偽物のラベル
criterion = nn.BCELoss() # バイナリ交差エントロピー

# 学習を行う関数
def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    # 学習モード
    netD.train()
    netG.train()

    for epoch in range(1, n_epochs+1):
        for x, _ in dataloader:
            x = x.to(device)

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            # 識別器の学習
            z = make_noise(batch_size) # ノイズを生成
            fake = netG(z) # 偽物を生成
            pred_fake = netD(fake) # 偽物を判定
            pred_real = netD(x) # 本物を判定
            loss_fake = criterion(pred_fake, fake_labels) # 偽物の判定に対する誤差
            loss_real = criterion(pred_real, real_labels) # 本物の判定に対する誤差
            lossD = loss_fake + loss_real # 二つの誤差の和
            lossD.backward() # 逆伝播
            optimD.step() # パラメータ更新

            # 生成器の学習
            fake = netG(z) # 偽物を生成
            pred = netD(fake) # 偽物を判定
            # lossG = pred.sum() # 和をとる
            lossG = criterion(pred, real_labels)
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')
        if epoch % write_interval == 0:
            write(netG,name="out") # 生成器で画像を生成し描画する


netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)

print('初期状態')
write(netG,name="out")
train(netD, netG, optimD, optimG, n_epochs)
