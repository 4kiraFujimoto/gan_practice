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

# #Discriminatorのノイズのつけ方パターン２（depois）
# class Discriminator(nn.Module):
#     def __init__(self, img_shape, num_classes):
#         super(Discriminator, self).__init__()
#         self.img_shape = img_shape
#         self.num_classes = num_classes

#         # ラベル埋め込みの定義
#         self.label_embedding = nn.Embedding(num_classes, int(np.prod(img_shape))) #10->784

#         # 連結された入力を扱うMLP
#         self.net = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.4),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.4),
#             nn.Linear(512, 1),
#             # nn.Sigmoid()
#         )

#     def forward(self, img, label):
#         label_embedding = self.label_embedding(label) # ラベルの埋め込みを取得
#         label_embedding = label_embedding.view(label_embedding.size(0), -1)
#         flat_img = img.view(img.size(0), -1) # 画像の平坦化
#         model_input = flat_img * label_embedding # 画像とラベル埋め込みの要素ごとの積
#         validity = self.net(model_input) # 判別器モデルを通す
#         return validity

class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        channels, height, width = self.img_shape  # 画像のチャンネル数、サイズを取得

        # ラベル埋め込みの定義
        self.label_embedding = nn.Embedding(num_classes, height * width)

        # 畳み込み層
        self.conv1 = nn.Conv2d(channels + 1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # 最終の全結合層
        self.fc = nn.Linear(4096, 1)  # 全結合層の入力次元を4096に修正

    def forward(self, img, label):
        # ラベルを埋め込みベクトルに変換
        label_embedding = self.label_embedding(label).view(label.size(0), 1, self.img_shape[1], self.img_shape[2])

        # 画像とラベルを結合（チャネル方向に結合）
        d_in = torch.cat((img, label_embedding), dim=1)

        # 畳み込み層の適用
        x = F.leaky_relu(self.bn1(self.conv1(d_in)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)

        # 出力のサイズを確認
        # print(f"Conv output size: {x.size()}")  # ここで出力サイズを確認

        # 平坦化して全結合層へ
        x = x.view(x.size(0), -1)  # バッチサイズに従って平坦化
        # print(x.shape)  # 平坦化後のサイズを確認
        validity = self.fc(x)
        validity = torch.sigmoid(validity)  

        return validity



# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             self._linear(nz, 256),
#             self._linear(256, 512),
#             self._linear(512, 1024),
#             nn.Linear(1024, image_size),
#             # nn.Tanh()
#             nn.Sigmoid()
#         )

#     def _linear(self, input_size, output_size):
#         return nn.Sequential(
#             nn.Linear(input_size, output_size),
#             nn.BatchNorm1d(output_size,momentum=0.8),
#             nn.ReLU(0.2)
#         )

#     def forward(self, x):
#         x = x.view(-1, nz)
#         y = self.net(x)
#         y = y.view(-1, 1, w, h) # 784 -> 1x28x28
#         return y

class Generator(nn.Module):
    def __init__(self, nz, img_shape):
        super(Generator, self).__init__()
        self.nz = nz  # 潜在変数の次元数
        self.img_shape = img_shape  # 画像の形状 (channels, height, width)

        # 全結合層で特徴抽出
        self.fc = nn.Sequential(
            nn.Linear(self.nz, 128 * (img_shape[1] // 4) * (img_shape[2] // 4)),
            nn.BatchNorm1d(128 * (img_shape[1] // 4) * (img_shape[2] // 4)),
            nn.ReLU(True)
        )

        # 畳み込み層
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2倍のサイズに拡張
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # さらに2倍のサイズに拡張
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, img_shape[0], kernel_size=3, stride=1, padding=1),  # チャンネル数を画像のチャンネル数に変換
            nn.Sigmoid()  # 出力を0から1に正規化
        )

    def forward(self, z):
        # 全結合層で特徴量を生成し、4次元テンソルに変換
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.img_shape[1] // 4, self.img_shape[2] // 4)

        # 畳み込み層で画像生成
        img = self.conv_blocks(out)
        return img


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
    global augmented_img_counter
    netG.eval()
    z = make_noise(torch.tensor([label]*numbers_of_images))
    gen_images = netG(z)
    gen_images = gen_images.view(numbers_of_images,1,28,28)
    save_dir = './cgan_generated_images'
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for i in range(numbers_of_images):
        img = to_pil(gen_images[i])  # TensorをPIL Imageに変換
        img.save(f"{save_dir}/generated_image_{augmented_img_counter}.png")  # 画像をPNG形式で保存
        augmented_img_counter += 1
        # print(image.shape)


# 画像読み込み用の関数
def load_generated_images(path):
    transform = transforms.ToTensor()  # PIL画像をTensorに変換
    images = []
    for img_path in glob.glob(f"{path}/*.png"):
        img = Image.open(img_path)  # 画像を開く
        img_tensor = transform(img)  # Tensorに変換
        images.append(img_tensor)
    return torch.stack(images)

# 間違ったラベルの生成
def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels


def train(dataloader, netD, netG, netL, optimD, optimG, optimL, n_epochs, write_interval=1):
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
    #最終のエポックのパラメータを取得する。パラメタの保存はもっと考える必要あり。
    torch.save(netG, "netG_parameter")
    torch.save(netD, "netD_parameter")
    




##main関数###
if __name__ == "__main__":
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
    img_shape=(1,28,28)
    # netD = Discriminator(img_shape=784, num_classes=10).to(device)
    netD = Discriminator(img_shape=(1,28,28), num_classes=10).to(device)
    netG = Generator(nz,img_shape=(1,28,28)).to(device)
    netL = LeNet().to(device) #authenticator
    optimD = optim.Adam(netD.parameters(), lr=0.0002)
    optimG = optim.Adam(netG.parameters(), lr=0.0002)
    optimL = optim.Adam(netL.parameters(), lr=0.0002)
    n_epochs = 100


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

    

    print('初期状態')
    write_0to9(netG)
    DO_TRAIN = False
    if DO_TRAIN:
        mnist_dataset = MNIST(root="./data", train=True, download=False, transform=transforms.ToTensor())
        
        # 3と7だけをフィルタリング
        def filter_mnist(dataset, labels_to_keep):
            indices = torch.where(torch.isin(dataset.targets, torch.tensor(labels_to_keep)))[0] #torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True).
            filtered_data = torch.utils.data.Subset(dataset, indices)
            return filtered_data
        
        # ラベル 3 と 7 だけを抽出
        # labels_to_keep = [0,1,2,3,4,5 ]
        labels_to_keep = [0,1,2,3,4,5,6,7,8,9]
        mnist_dataset = filter_mnist(mnist_dataset, labels_to_keep)

        # サブセットからデータとラベルを取得
        data = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
        labels = torch.tensor([mnist_dataset[i][1] for i in range(len(mnist_dataset))])

        # データとラベルをシャッフル
        perm = torch.randperm(len(data))
        data = data[perm]
        labels = labels[perm]

        # 末尾10%を攻撃用に分離
        num_samples = len(labels)
        attack_split_idx = int(0.9 * num_samples)

        # 最初の90%と末尾10%を分離
        data, for_attack_data = data[:attack_split_idx], data[attack_split_idx:]
        labels, for_attack_labels = labels[:attack_split_idx], labels[attack_split_idx:]

        # # 末尾10%のラベルを反転
        # attack_labels = torch.where(attack_labels == 3, 4, 3)

        # 最初の90%を50%と40%に分離 (40%をクリーンデータとする。)
        num_samples = len(labels)
        clean_split_idx = int(0.5 * num_samples)
        # clean_split_idx = 1
        
        data, clean_data = data[:clean_split_idx], data[clean_split_idx : attack_split_idx]
        labels, clean_labels = labels[:clean_split_idx], labels[clean_split_idx : attack_split_idx]


        _ = torch.utils.data.TensorDataset(data, labels)
        clean_dataset = torch.utils.data.TensorDataset(clean_data, clean_labels)
        for_attack_dataset = torch.utils.data.TensorDataset(for_attack_data, for_attack_labels)
        clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for_attack_dataloader = DataLoader(for_attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        train(clean_dataloader, netD, netG, netL, optimD, optimG, optimL, n_epochs)
        print("netD, netGのパラメータを保存しました。(ファイル名：netD_parameter, netG_parameter)")
        # write_specific_number(netG, numbers_of_images=1000, label=0)
        
    else:
        numbers_of_images = 20000
        netG = torch.load("netG_parameter")
        netD = torch.load("netD_parameter")
        
        # write_0to9(netG)
        # MNISTのデータセットをロード
        mnist_dataset = MNIST(root="./data", train=True, download=False, transform=transforms.ToTensor())
        
        # 3と7だけをフィルタリング
        def filter_mnist(dataset, labels_to_keep):
            indices = torch.where(torch.isin(dataset.targets, torch.tensor(labels_to_keep)))[0] #torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True).
            # filtered_data = torch.utils.data.Subset(dataset, indices)
            dataset.targets = dataset.targets[indices]  # ラベルをフィルタリング
            dataset.data = dataset.data[indices]        # データをフィルタリング
            return dataset
            return filtered_data
        
        # ラベル 3 と 7 だけを抽出
        labels_to_keep = [3, 7]
        mnist_dataset = filter_mnist(mnist_dataset, labels_to_keep)
        ### test_filtered = filter_mnist(mnist_test, labels_to_keep)
        # data = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
        # labels = torch.tensor([mnist_dataset[i][1] for i in range(len(mnist_dataset))])
        # num_samples = len(labels)
        # clean_split_idx = int(0.5 * num_samples)
        # _, clean_data = data[:clean_split_idx], data[clean_split_idx : attack_split_idx]
        # _, clean_labels = labels[:clean_split_idx], labels[clean_split_idx : attack_split_idx]



        #データ生成
        augmented_img_counter = 0
        write_specific_number(netG, numbers_of_images=numbers_of_images//2, label=3)
        write_specific_number(netG, numbers_of_images=numbers_of_images//2, label=7)


        # 保存された画像をロード
        loaded_images = load_generated_images('./cgan_generated_images')
        # print(f"loaded_imagesの形状：{loaded_images.shape}")
        
        # MNISTの画像と生成した画像を結合
        mnist_images, mnist_labels = mnist_dataset.data.float(), mnist_dataset.targets
        # print(f"mnist_imagesの形状：{mnist_images.shape}")
        # print(f"mnist_images.unsqueezeの形状：{mnist_images.unsqueeze(1).shape}")
        combined_images = torch.cat((mnist_images.unsqueeze(1), loaded_images), 0)  # 画像を結合
        # print(f"combined_imagesの形状：{combined_images.shape}")

        # 生成画像のラベルを3と7に設定
        generated_labels = torch.cat((
            torch.full((numbers_of_images//2,), 3),  # 最初の半分はラベル3
            torch.full((numbers_of_images//2,), 7)   # 残りはラベル7
        ))
        combined_labels = torch.cat((mnist_labels, generated_labels), 0) # 生成画像のラベルを"0"とする
        # print(combined_labels[-10:])


        ##combined_imagesとcombined_labelsを保存して、wgangpで呼び出せるようにする。
        #データセットを結合後のものに更新
        combined_dataset = TensorDataset(combined_images, combined_labels)
        # combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)    
        
        print(type(combined_labels))
        combined_labels = combined_labels.long()
        print(torch.unique(combined_labels))  # ユニークなラベル値を確認

        #生成したデータセットをセーブ
        torch.save(combined_dataset, "Synthetic_Dataset.pt")
        


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
    
    # create_gif(in_dir='/home/fujimoto-a/research/ganpra/cgan_images', out_filename='/home/fujimoto-a/research/ganpra/cgan_images_gif/animation.gif') # GIFアニメーションを作成する関数を実行する











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