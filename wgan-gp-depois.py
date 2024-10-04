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
from scipy import stats
import matplotlib.pyplot as plt

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
            # nn.Sigmoid()
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


def make_noise(labels):
    labels = eye[labels].to(device)
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
    
    file_path = f"/home/fujimoto-a/research/ganpra/wgan_images/wgan{img_counter}.jpg"
    img.save(file_path)
    img_counter += 1

# 間違ったラベルの生成
def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels

def gradient_penalty(netD, real_images, fake_images, labels, epsilon=0.5):
        interpolates = epsilon * real_images + (1 - epsilon) * fake_images
        interpolates.requires_grad_(True)

        critic_interpolates = netD(interpolates,labels)
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size()).to(real_images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    

def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    # 学習モード
    
    netD.train()
    netG.train()


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
            # print(z.shape)  
            fake = netG(z) # 偽物を生成       
            critic_fake = netD(fake, labels) # 偽物を判定
            critic_real_true = netD(X, labels) # 本物&正しいラベルを判定
            critic_real_false = netD(X, false_labels) # 本物&間違ったラベルを判定

            # 誤差を計算
            # loss_fake = criterion(pred_fake, fake_labels)
            # loss_real_true = criterion(pred_real_true, real_labels)
            # loss_real_false = criterion(pred_real_false, fake_labels)
            loss_fake = critic_fake.mean()
            loss_real_false = critic_real_false.mean()
            loss_real_true = critic_real_true.mean()
            gp = gradient_penalty(netD, X, fake, labels,epsilon=0.5)
            GRADIENT_PENALTY_WEIGHT = 10
            lossD = loss_fake +loss_real_false - loss_real_true + GRADIENT_PENALTY_WEIGHT * gp # 全ての和をとる        
            lossD.backward() # 逆伝播
            optimD.step() # パラメータ更新

            #------------------
            # Generatorの学習
            #------------------
            fake = netG(z) # 偽物を生成
            pred = netD(fake, labels) # 偽物を判定
            # lossG = criterion(pred, real_labels) # 誤差を計算
            lossG = -pred.mean() 
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')
        if write_interval and epoch % write_interval == 0:
            write(netG)

    #最終のエポックのパラメータを取得する。パラメタの保存はもっと考える必要あり。
    torch.save(netG, "wgan_netG_parameter")
    torch.save(netD, "wgan_netD_parameter")



##main関数###

# faulthandler.enable()
batch_size = 64
nz = 100
noise_std = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
#### dataset = MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )

#### dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=True
# )

# 生成したデータセットのロード
synthetic_dataset = torch.load("Synthetic_Dataset.pt")

# synthetic_datasetはTensorDataset
images, labels = synthetic_dataset.tensors
# ラベルを3を0に、7を1に変換する #これをやらないと、make_noiseのlabels = eye[labels].to(device)でエラー
replace_labels = labels.clone()  # 元のラベルをコピー
replace_labels[labels == 3] = 0  # 3を0に変換
replace_labels[labels == 7] = 1  # 7を1に変換
# 新しいデータセットを作成
new_dataset = TensorDataset(images, replace_labels)

# データローダーの作成
dataloader = DataLoader(
    new_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
#### n_classes = len(torch.unique(dataset.targets)) # 10
n_classes = len(torch.unique(new_dataset.tensors[1]))  # ラベルのユニーク値を数える
print(f"合成データのラベルのセット = {torch.unique(new_dataset.tensors[1])}")
print(f"合成データのラベルの数(n_classes) = {n_classes}")
sample_x, _ = next(iter(dataloader))
w, h = sample_x.shape[-2:]                     # (28, 28)
image_size = w * h                             # 784

# n_classes =10
eye = torch.eye(n_classes, device=device)
# z = make_noise(torch.tensor([3, 7], device=device))


fake_labels = torch.zeros(batch_size, 1).to(device) # 偽物のラベル
real_labels = torch.ones(batch_size, 1).to(device) # 本物のラベル
criterion = nn.BCELoss() # バイナリ交差エントロピー BCEと違って、CrossEntropyLossは内部でsigmoid関数がかけられるので、生データそのまま入れれる。


netD = Discriminator(img_shape=784, num_classes=10).to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)

n_epochs = 100


##画像用
folder_name = "wgan_images"# フォルダ名
current_directory = os.getcwd()# 現在のディレクトリを取得
folder_path = os.path.join(current_directory, folder_name)# フォルダのパスを作成
if not os.path.exists(folder_path):# フォルダが存在しない場合は作成する
    os.makedirs(folder_path)
    print(f"'{folder_name}' フォルダを作成しました。")
else:
    print(f"'{folder_name}' フォルダはすでに存在します。")
img_counter = 0 # グローバルカウンタ（初期値は0）


# DO_TRAIN = True #step2(wgan訓練)
DO_TRAIN = False #step3(検査)
if DO_TRAIN:
    print('初期状態')
    write(netG)
    train(netD, netG, optimD, optimG, n_epochs)
else: #検査
    # netG = torch.load("wgan_netG_parameter")
    netD = torch.load("wgan_netD_parameter")
    netD.eval()
    critic_validities = []
    for X, labels in dataloader:
        X = X.to(device)
        labels = labels.to(device) 
        critic_validity = netD(X, labels)
        critic_validities.append(critic_validity.view(-1).detach().cpu().numpy())

    critic_validities = np.concatenate(critic_validities)
    # ヒストグラムのプロット
    plt.hist(critic_validities, bins=30, alpha=0.75, color='blue')
    plt.title('Histogram of Validity Outputs')
    plt.xlabel('Validity')
    plt.ylabel('Frequency')
    plt.savefig("validity_hist.png") 




    ###攻撃作成###
    mnist_dataset = MNIST(root="./data", train=True, download=False, transform=transforms.ToTensor())
    # 3と7だけをフィルタリング
    def filter_mnist(dataset, labels_to_keep):
        indices = torch.where(torch.isin(dataset.targets, torch.tensor(labels_to_keep)))[0] #torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True).
        filtered_data = torch.utils.data.Subset(dataset, indices)
        return filtered_data
    # ラベル 3 と 7 だけを抽出
    # labels_to_keep = [0,1,2,3,4,5 ]
    labels_to_keep = [3,7]
    mnist_dataset = filter_mnist(mnist_dataset, labels_to_keep)
    # サブセットからデータとラベルを取得
    data = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
    labels = torch.tensor([mnist_dataset[i][1] for i in range(len(mnist_dataset))])
    # データとラベルをシャッフル
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]
    attack_dataset = torch.utils.data.TensorDataset(data, labels)
    attatck_images, attack_labels = attack_dataset.tensors
    # ラベルを3を0に、7を1に変換する #これをやらないと、make_noiseのlabels = eye[labels].to(device)でエラー
    replace_attack_labels = attack_labels.clone()  # 元のラベルをコピー
    replace_attack_labels[attack_labels == 3] = 1  # 3を0に変換
    replace_attack_labels[attack_labels == 7] = 0  # 7を1に変換
    # 新しいデータセットを作成
    attack_dataset = TensorDataset(attatck_images, replace_attack_labels)
    attack_dataloader = DataLoader(
        attack_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    netD.eval()
    attack_critic_validities = []
    for X, labels in attack_dataloader:
        X = X.to(device)
        labels = labels.to(device) 
        critic_validity = netD(X, labels)
        attack_critic_validities.append(critic_validity.view(-1).detach().cpu().numpy())
    attack_critic_validities = np.concatenate(attack_critic_validities)

    # ヒストグラムのプロット
    plt.figure(figsize=(10, 6))
    # plt.hist(critic_validities, bins=30, color='blue', alpha=0.5, label='Critic Validities')
    plt.hist(attack_critic_validities, bins=30, color='red', alpha=0.5, label='Attack Critic Validities')
    # グラフの装飾
    plt.title('Histogram of Critic Validities')
    plt.xlabel('Validity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig("validity_attack.png") 

    # ヒストグラムのプロット
    plt.figure(figsize=(10, 6))
    plt.hist(critic_validities, bins=30, color='blue', alpha=0.5, label='Critic Validities')
    plt.hist(attack_critic_validities, bins=30, color='red', alpha=0.5, label='Attack Critic Validities')
    # グラフの装飾
    plt.title('Histogram of Critic Validities')
    plt.xlabel('Validity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig("validity_comparison.png") 


    ##正規分布を仮定した際の下位5%の閾値計算
    # validityの平均と標準偏差を計算
    mean_validity = np.mean(critic_validities)
    std_validity = np.std(critic_validities)

    # 下位5%に相当する閾値を計算
    threshold_5_percent = stats.norm.ppf(0.05, loc=mean_validity, scale=std_validity)

    print(f"下位5%に相当する閾値: {threshold_5_percent}")