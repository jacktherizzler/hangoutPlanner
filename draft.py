import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import os

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        mask = torch.zeros_like(image)
        w, h = image.size
        mask_x, mask_y = torch.randint(0, w // 2, (1,)).item(), torch.randint(0, h // 2, (1,)).item()
        mask_width, mask_height = torch.randint(w // 8, w // 4, (1,)).item(), torch.randint(h // 8, h // 4, (1,)).item()
        mask.paste(0, (mask_x, mask_y, mask_x + mask_width, mask_y + mask_height))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        corrupted_image = image * (1 - mask)
        return corrupted_image, image, mask

class InpaintingModel(nn.Module):
    def __init__(self):
        super(InpaintingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = InpaintingDataset('path/to/your/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = InpaintingModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for corrupted, original, mask in dataloader:
        corrupted = corrupted.cuda()
        original = original.cuda()
        mask = mask.cuda()
        output = model(corrupted)
        loss = criterion(output, original)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'inpainting_model.pth')
