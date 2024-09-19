import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import display
import faulthandler
import os
import numpy as np
import torch.nn.functional as F
import glob
from PIL import Image

#Discriminatorのノイズのつけ方パターン２（depois）
class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        # ラベル埋め込みの定義
        self.label_embedding = nn.Embedding(num_classes, int(np.prod(img_shape))) #10->784

        # 連結された入力を扱うMLP
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        label_embedding = self.label_embedding(label) # ラベルの埋め込みを取得
        label_embedding = label_embedding.view(label_embedding.size(0), -1)
        flat_img = img.view(img.size(0), -1) # 画像の平坦化
        model_input = flat_img * label_embedding # 画像とラベル埋め込みの要素ごとの積
        validity = self.net(model_input) # 判別器モデルを通す
        return validity



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear(nz, 256),
            self._linear(256, 512),
            self._linear(512, 1024),
            nn.Linear(1024, image_size),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size,momentum=0.8),
            nn.ReLU(0.2)
        )

    def forward(self, x):
        x = x.view(-1, nz)
        y = self.net(x)
        y = y.view(-1, 1, w, h) # 784 -> 1x28x28
        return y


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.net = nn.Sequential(
#             # 第一の畳み込み層、ReLU活性化、最大プーリング
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # 第二の畳み込み層、ReLU活性化、最大プーリング
#             nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Flatten層（Sequential内でのFlattenはできないため、次の全結合層で行う）
#             nn.Flatten(),

#             # 全結合層、ReLU活性化
#             nn.Linear(50 * 7 * 7, 500),
#             nn.ReLU(),

#             # 出力層（ソフトマックスはこの後の損失関数で適用）
#             nn.Linear(500, 10)
#         )

#     def forward(self, x):
#         y = self.net(x)
#         # y = y.view(-1,10)
#         return y
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 入力チャネル1、出力チャネル6、フィルタサイズ5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 入力チャネル6、出力チャネル16、フィルタサイズ5x5
        self.fc1 = nn.Linear(16*4*4, 120)  # 全結合層、入力サイズ16*4*4、出力サイズ120
        self.fc2 = nn.Linear(120, 84)      # 全結合層、入力サイズ120、出力サイズ84
        self.fc3 = nn.Linear(84, 10)       # 全結合層、入力サイズ84、出力サイズ10

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 畳み込み+ReLU
        x = F.max_pool2d(x, 2)         # プーリング層
        x = F.relu(self.conv2(x))      # 畳み込み+ReLU
        x = F.max_pool2d(x, 2)         # プーリング層
        x = x.view(-1, 16*4*4)         # 平坦化
        x = F.relu(self.fc1(x))        # 全結合層+ReLU
        x = F.relu(self.fc2(x))        # 全結合層+ReLU
        x = self.fc3(x)                # 最終出力層
        return x



def make_noise(labels):
    labels = eye[labels].to(device)
    labels = labels.repeat_interleave(nz // n_classes, dim=1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    return z

# 画像描画
def write_0to9(netG, n_rows=1, size=64):
    netG.eval()
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

def write_specific_number(netG, numbers_of_images:int = 1000, label:int = 0 ) -> None:
    netG.eval()
    z = make_noise(torch.tensor([label]*numbers_of_images))
    gen_images = netG(z)
    gen_images = gen_images.view(numbers_of_images,1,28,28)
    save_dir = './cgan_generated_images'
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for i in range(numbers_of_images):
        img = to_pil(gen_images[i])  # TensorをPIL Imageに変換
        img.save(f"{save_dir}/generated_image_{i}.png")  # 画像をPNG形式で保存
        # print(image.shape)



# 間違ったラベルの生成
def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels


def train(netD, netG, netL, optimD, optimG, optimL, n_epochs, write_interval=1):
    # 学習モード
    netD.train()
    netG.train()
    netL.train()

    for epoch in range(1, n_epochs+1):
        for X, labels in dataloader:
            X = X.to(device) # 本物の画像
            # print(X.shape)
            # print(labels.shape)
            labels = labels.to(device) # 正しいラベル
            false_labels = make_false_labels(labels) # 間違ったラベル

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            #---------------------
            # Discriminatorの学習
            #--------------------- 
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
            #authenticatorの誤差を足し算#ラベルをonehotして合成して、lenet学習させる。誤差をlossDに足す。
            lenet_X = torch.cat((X, fake), dim=0) #authenticator用の学習データ
            # one_hot_labels = F.one_hot(labels, num_classes=10)
            lenet_label = torch.cat((labels, labels), dim=0) #authenticator用の学習ラベル
            lenet_pred = netL(lenet_X)
            lenet_loss = nn.CrossEntropyLoss()(lenet_pred, lenet_label)
            lossD += lenet_loss            
            lossD.backward() # 逆伝播
            optimD.step() # パラメータ更新「

            #------------------
            # Generatorの学習
            #------------------
            fake = netG(z) # 偽物を生成
            pred = netD(fake, labels) # 偽物を判定
            lossG = criterion(pred, real_labels) # 誤差を計算
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')
        if write_interval and epoch % write_interval == 0:
            write_0to9(netG)

    #preserve model parameter
    torch.save(netG, "netG_parameter")
    torch.save(netD, "netD_parameter")
    




##main関数###

# faulthandler.enable()
batch_size = 64
nz = 100
noise_std = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"device = {device}")

dataset = MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

sample_x, _ = next(iter(dataloader))
n_classes = len(torch.unique(dataset.targets)) # 10
w, h = sample_x.shape[-2:]                     # (28, 28)
image_size = w * h                             # 784

eye = torch.eye(n_classes, device=device)

fake_labels = torch.zeros(batch_size, 1).to(device) # 偽物のラベル
real_labels = torch.ones(batch_size, 1).to(device) # 本物のラベル
criterion = nn.BCELoss() # バイナリ交差エントロピー BCEと違って、CrossEntropyLossは内部でsigmoid関数がかけられるので、生データそのまま入れれる。

netD = Discriminator(img_shape=784, num_classes=10).to(device)
netG = Generator().to(device)
netL = LeNet().to(device) #authenticator
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
optimL = optim.Adam(netL.parameters(), lr=0.0002)
n_epochs = 5


##画像用
folder_name = "cgan_images"# フォルダ名
current_directory = os.getcwd()# 現在のディレクトリを取得
folder_path = os.path.join(current_directory, folder_name)# フォルダのパスを作成
if not os.path.exists(folder_path):# フォルダが存在しない場合は作成する
    os.makedirs(folder_path)
    print(f"'{folder_name}' フォルダを作成しました。")
else:
    print(f"'{folder_name}' フォルダはすでに存在します。")
img_counter = 0 # グローバルカウンタ（初期値は0）

# 画像読み込み用の関数
def load_generated_images(path):
    transform = transforms.ToTensor()  # PIL画像をTensorに変換
    images = []
    for img_path in glob.glob(f"{path}/*.png"):
        img = Image.open(img_path)  # 画像を開く
        img_tensor = transform(img)  # Tensorに変換
        images.append(img_tensor)
    return torch.stack(images)

print('初期状態')
write_0to9(netG)
DO_TRAIN = False
if DO_TRAIN:
    train(netD, netG, netL, optimD, optimG, optimL, n_epochs)
    write_specific_number(netG, numbers_of_images=1000, label=0)

else:
    numbers_of_images = 1000
    netG = torch.load("netG_parameter")
    netD = torch.load("netD_parameter")
    write_0to9(netG)
    write_specific_number(netG, numbers_of_images=1000, label=0)
    # 保存された画像をロード
    loaded_images = load_generated_images('./cgan_generated_images')
    print(f"loaded_imagesの形状：{loaded_images.shape}")

    # MNISTのデータセットをロード
    mnist_dataset = MNIST(root="./data", train=True, download=False, transform=transforms.ToTensor())
    # MNISTの画像と生成した画像を結合
    mnist_images, mnist_labels = mnist_dataset.data.float(), mnist_dataset.targets
    print(f"mnist_imagesの形状：{mnist_images.shape}")
    print(f"mnist_images.unsqueezeの形状：{mnist_images.unsqueeze(1).shape}")
    combined_images = torch.cat((mnist_images.unsqueeze(1), loaded_images), 0)  # 画像を結合
    print(f"combined_imagesの形状：{combined_images.shape}")
    combined_labels = torch.cat((mnist_labels, torch.full((numbers_of_images,), 0)))  # 生成画像のラベルを"10"とする

    # データセットを結合後のものに更新
    combined_dataset = TensorDataset(combined_images, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)    
    

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
 
create_gif(in_dir='/home/fujimoto-a/research/ganpra/cgan_images', out_filename='/home/fujimoto-a/research/ganpra/cgan_images_gif/animation.gif') # GIFアニメーションを作成する関数を実行する


##test lenet
# # 損失関数とオプティマイザの定義
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(netL.parameters(), lr=0.01, momentum=0.9)
# # ダミーデータの作成
# dummy_input = torch.randn(4, 1, 28, 28).to(device)  # バッチサイズ4、チャンネル1、28x28ピクセルのダミーデータ
# dummy_labels = torch.randint(0, 10, (4,)).to(device)  # バッチサイズ4のダミーラベル（0から9のクラス）

# # 学習ループ
# num_epochs = 10
# for epoch in range(num_epochs):
#     netL.train()  # 学習モード
#     optimizer.zero_grad()  # 勾配の初期化

#     # フォワードプロパゲーション
#     outputs = netL(dummy_input)
#     loss = criterion(outputs, dummy_labels)

#     # バックプロパゲーション
#     loss.backward()
#     optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')