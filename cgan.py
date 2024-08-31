import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import display
import faulthandler
import os
import glob
from PIL import Image

#Discriminatorのノイズのつけ方パターン1（qiita）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._eye = torch.eye(n_classes, device=device) # 条件ベクトル生成用の単位行列

    def forward(self, x, labels):
        labels = self._eye[labels] # ラベル(条件)をone-hotベクトルに
        x = x.view(batch_size, -1) # 画像を1次元に
        x = torch.cat([x, labels], dim=1) # 画像と条件ベクトルを結合
        y = self.net(x)
        return y

#これを使うとがびがび　→ ranhは-1~1にスケールするからや。
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear(nz, 256),
            self._linear(256, 512),
            self._linear(512, 1024),
            nn.Linear(1024, image_size),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size,momentum=0.8),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, nz)
        y = self.net(x)
        y = y.view(-1, 1, w, h) # 784 -> 1x28x28
        return y

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             self._linear(nz, 128),
#             self._linear(128, 256),
#             self._linear(256, 512),
#             nn.Linear(512, image_size),
#             nn.Sigmoid()
#         )

#     def _linear(self, input_size, output_size):
#         return nn.Sequential(
#             nn.Linear(input_size, output_size),
#             nn.BatchNorm1d(output_size),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = x.view(-1, nz)
#         y = self.net(x)
#         y = y.view(-1, 1, w, h) # 784 -> 1x28x28
#         return y



def make_noise(labels):
    labels = eye[labels]
    labels = labels.repeat_interleave(nz // n_classes, dim=1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    return z

# 画像描画
def write(netG, n_rows=1, size=64):
    global img_counter
    n_images = n_rows * n_classes
    z = make_noise(torch.tensor(list(range(n_classes)) * n_rows))
    images = netG(z)
    images = transforms.Resize(size)(images)
    img = torchvision.utils.make_grid(images, n_images // n_rows)
    img = transforms.functional.to_pil_image(img)
    
    file_path = f"/home/fujimoto-a/research/ganpra/cgan_images/cgan{img_counter}.jpg"
    img.save(file_path)
    img_counter += 1

# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    imgs = []                                                   # 画像をappendするための空配列を定義
 
    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        imgs.append(img)                                        # 画像をappendで配列に格納していく
 
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)
 

# 間違ったラベルの生成
def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels



def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    # 学習モード
    netD.train()
    netG.train()

    for epoch in range(1, n_epochs+1):
        for X, labels in dataloader:
            X = X.to(device) # 本物の画像 0~1スケールです。
            labels = labels.to(device) # 正しいラベル
            false_labels = make_false_labels(labels) # 間違ったラベル

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            # Discriminatorの学習
            z = make_noise(labels) # ノイズを生成
            fake = netG(z) # 偽物を生成
            pred_fake = netD(fake, labels) # 偽物を判定
            pred_real_true = netD(X, labels) # 本物&正しいラベルを判定
            pred_real_false = netD(X, false_labels) # 本物&間違ったラベルを判定
            # 誤差を計算
            loss_fake = criterion(pred_fake, fake_labels)
            loss_real_true = criterion(pred_real_true, real_labels)
            loss_real_false = criterion(pred_real_false, fake_labels)
            lossD = loss_fake + loss_real_true + loss_real_false # 全ての和をとる
            lossD.backward() # 逆伝播
            optimD.step() # パラメータ更新

            # Generatorの学習
            fake = netG(z) # 偽物を生成
            pred = netD(fake, labels) # 偽物を判定
            lossG = criterion(pred, real_labels) # 誤差を計算
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')
        if write_interval and epoch % write_interval == 0:
            write(netG)


##main関数###

# faulthandler.enable()
batch_size = 64
nz = 100
noise_std = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
    drop_last=True
)

sample_x, _ = next(iter(dataloader))
n_classes = len(torch.unique(dataset.targets)) # 10
w, h = sample_x.shape[-2:]                     # (28, 28)
image_size = w * h                             # 784


eye = torch.eye(n_classes, device=device)


fake_labels = torch.zeros(batch_size, 1).to(device) # 偽物のラベル
real_labels = torch.ones(batch_size, 1).to(device) # 本物のラベル
criterion = nn.BCELoss() # バイナリ交差エントロピー


netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
n_epochs = 10


##画像用
folder_name = "cgan_images"# フォルダ名
current_directory = os.getcwd()# 現在のディレクトリを取得
folder_path = os.path.join(current_directory, folder_name)# フォルダのパスを作成
if not os.path.exists(folder_path):# フォルダが存在しない場合は作成する
    os.makedirs(folder_path)
    print(f"'{folder_name}' フォルダを作成しました。")
# else:
#     print(f"'{folder_name}' フォルダはすでに存在します。")
img_counter = 0 # グローバルカウンタ（初期値は0）


print('初期状態')
# write(netG)
# train(netD, netG, optimD, optimG, n_epochs)
create_gif(in_dir='/home/fujimoto-a/research/ganpra/cgan_images', out_filename='/home/fujimoto-a/research/ganpra/cgan_images_gif/animation.gif') # GIFアニメーションを作成する関数を実行する