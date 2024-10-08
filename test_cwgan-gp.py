import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).expand_as(real_samples).to(device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples).to(device)
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, labels)
    fake = torch.ones(real_samples.size(0), 1, requires_grad=False).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    

import matplotlib.pyplot as plt

def save_generated_images(generator, epoch, n_images=10):
    z = torch.randn(n_images, 100).to(device)
    labels = torch.arange(0, 10).to(device)
    
    with torch.no_grad():
        gen_images = generator(z, labels)
        gen_images = gen_images * 0.5 + 0.5  # Rescale to [0, 1]

    # Plot and save images
    fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
    for i in range(n_images):
        axes[i].imshow(gen_images[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close(fig) 



mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

generator = Generator(z_dim=100, num_classes=10, img_size=28).to(device)
discriminator = Discriminator(num_classes=10, img_size=(1, 28, 28)).to(device)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
lambda_gp = 10
epochs = 30

def train(discriminator, generator, optimizer_D, optimizer_G, epochs):
    for epoch in range(epochs):
        print(f"epoch : {epoch}")
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(imgs.size(0), 100).to(device)
            gen_labels = torch.randint(0, 10, (imgs.size(0),)).to(device)
            fake_imgs = generator(z, gen_labels).detach()
            
            real_validity = discriminator(imgs, labels)
            fake_validity = discriminator(fake_imgs, gen_labels)
            gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data, labels.data)
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty)
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            
            gen_imgs = generator(z, gen_labels)
            g_loss = -torch.mean(discriminator(gen_imgs, gen_labels))
            g_loss.backward()
            optimizer_G.step()
        print(f"Epoch [{epoch}/{epochs}] - Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")  
        # Save generated images for the final epoch
        save_generated_images(generator, epoch)

train(discriminator, generator, optimizer_D, optimizer_G, epochs)
