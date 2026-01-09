import torch
import torch.nn as nn
from src.data_setup import create_dataloaders
from src.model import CNN
from src.engine import train_model, init_wandb_run
from src.utils import set_seed
import os
import random
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    set_seed()
    
    DATA_PATH = "./data/pokemon_clean"
    
   # 1. Accessing config values
    print(f"Training on: {cfg.device}")
    
    # 2. Setup Data (Accessing nested values)
    train_dl, val_dl, _ = create_dataloaders(
        clean_data_path=DATA_PATH,
        batch_size=cfg.training.batch_size
    )

    # 3. Initialize Model
    model = CNN(
        n_layers=cfg.model.n_layers,
        n_filters=list(cfg.model.n_filters), # Hydra uses ListConfig, convert to list
        dropout_rate=cfg.model.dropout_rate,
        num_classes=cfg.training.num_classes,
        kernel_sizes=cfg.model.kernel_sizes,
        fc_size=cfg.model.fc_size
    ).to(cfg.device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.training.lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3, 
                verbose=True
            )

    # 4. Init W&B using the Hydra config
    wandb_run = init_wandb_run(config=cfg)

    # 5. Start Training
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        device=cfg.device,             
        n_epochs=cfg.training.epochs,  
        wandb_run=wandb_run
    )
  

if __name__ == "__main__":
    main()