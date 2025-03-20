import os
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Our separate modules
from config import (DEVICE, STYLE_DIR, CONTENT_DIR, IMG_SIZE,
                    BATCH_SIZE, EPOCHS, LR, BETA1, BETA2,
                    LAMBDA_CYCLE, LAMBDA_ID, WEIGHTS_DIR)

from dataset import ChinesePaintingsDataset, get_transforms
from models import GeneratorResNet, Discriminator
from train_utils import weights_init_normal

def main():
    # 1. Dataset & DataLoader
    transform = get_transforms()
    train_dataset = ChinesePaintingsDataset(
        style_dir=STYLE_DIR,
        content_dir=CONTENT_DIR,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize generators and discriminators
    G_style2content = GeneratorResNet().to(DEVICE)  # G: style -> content
    G_content2style = GeneratorResNet().to(DEVICE)  # F: content -> style
    D_style = Discriminator().to(DEVICE)            # D_style
    D_content = Discriminator().to(DEVICE)          # D_content

    # 3. Initialize weights
    G_style2content.apply(weights_init_normal)
    G_content2style.apply(weights_init_normal)
    D_style.apply(weights_init_normal)
    D_content.apply(weights_init_normal)

    # 4. Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # 5. Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_style2content.parameters(), G_content2style.parameters()),
        lr=LR, betas=(BETA1, BETA2)
    )
    optimizer_D_style = torch.optim.Adam(D_style.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D_content = torch.optim.Adam(D_content.parameters(), lr=LR, betas=(BETA1, BETA2))

    # 6. Training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch}/{EPOCHS}]")
        for i, (style_img, content_img) in enumerate(tqdm(train_loader, leave=False)):
            style_img = style_img.to(DEVICE)
            content_img = content_img.to(DEVICE)

            ################################################################
            #  Train Generators (G_style2content & G_content2style)
            ################################################################
            optimizer_G.zero_grad()

            # Identity losses
            id_style = G_content2style(style_img)
            loss_id_style = criterion_identity(id_style, style_img) * LAMBDA_ID

            id_content = G_style2content(content_img)
            loss_id_content = criterion_identity(id_content, content_img) * LAMBDA_ID

            # GAN losses
            fake_content = G_style2content(style_img)
            pred_fake_content = D_content(fake_content)
            valid_label = torch.ones_like(pred_fake_content).to(DEVICE)
            loss_G_style2content = criterion_GAN(pred_fake_content, valid_label)

            fake_style = G_content2style(content_img)
            pred_fake_style = D_style(fake_style)
            loss_G_content2style = criterion_GAN(pred_fake_style, valid_label)

            # Cycle consistency losses
            rec_style = G_content2style(fake_content)
            loss_cycle_style = criterion_cycle(rec_style, style_img) * LAMBDA_CYCLE

            rec_content = G_style2content(fake_style)
            loss_cycle_content = criterion_cycle(rec_content, content_img) * LAMBDA_CYCLE

            # Total generator loss
            loss_G = (
                loss_G_style2content + loss_G_content2style
                + loss_cycle_style + loss_cycle_content
                + loss_id_style + loss_id_content
            )
            loss_G.backward()
            optimizer_G.step()

            ################################################################
            #  Train Discriminator: D_content
            ################################################################
            optimizer_D_content.zero_grad()
            # Real
            pred_real_content = D_content(content_img)
            valid_label = torch.ones_like(pred_real_content).to(DEVICE)
            loss_D_real = criterion_GAN(pred_real_content, valid_label)

            # Fake
            fake_content_detached = fake_content.detach()
            pred_fake_content = D_content(fake_content_detached)
            fake_label = torch.zeros_like(pred_fake_content).to(DEVICE)
            loss_D_fake = criterion_GAN(pred_fake_content, fake_label)

            loss_D_content_total = 0.5 * (loss_D_real + loss_D_fake)
            loss_D_content_total.backward()
            optimizer_D_content.step()

            ################################################################
            #  Train Discriminator: D_style
            ################################################################
            optimizer_D_style.zero_grad()
            # Real
            pred_real_style = D_style(style_img)
            valid_label = torch.ones_like(pred_real_style).to(DEVICE)
            loss_D_real = criterion_GAN(pred_real_style, valid_label)

            # Fake
            fake_style_detached = fake_style.detach()
            pred_fake_style = D_style(fake_style_detached)
            fake_label = torch.zeros_like(pred_fake_style).to(DEVICE)
            loss_D_fake = criterion_GAN(pred_fake_style, fake_label)

            loss_D_style_total = 0.5 * (loss_D_real + loss_D_fake)
            loss_D_style_total.backward()
            optimizer_D_style.step()

            # Print progress every N batches
            if i % 5 == 0:
                print(
                    f"Batch {i}/{len(train_loader)} | "
                    f"Loss_G: {loss_G.item():.4f} | "
                    f"Loss_D_content: {loss_D_content_total.item():.4f} | "
                    f"Loss_D_style: {loss_D_style_total.item():.4f}"
                )

        # 7. Save model checkpoints every 5 epochs
        if epoch % 5 == 0:
            torch.save(G_style2content.state_dict(), os.path.join(WEIGHTS_DIR, f"G_style2content_{epoch}.pth"))
            torch.save(G_content2style.state_dict(), os.path.join(WEIGHTS_DIR, f"G_content2style_{epoch}.pth"))
            torch.save(D_style.state_dict(),        os.path.join(WEIGHTS_DIR, f"D_style_{epoch}.pth"))
            torch.save(D_content.state_dict(),      os.path.join(WEIGHTS_DIR, f"D_content_{epoch}.pth"))

if __name__ == "__main__":
    main()
