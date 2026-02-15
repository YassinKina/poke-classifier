import torch
import torch.nn as nn
import os
from src import create_dataloaders
from src import DynamicCNN
from src import train_model, init_wandb_run
from src import set_seed, create_data_dir
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Orchestrates the end-to-end training pipeline for the Pokémon CNN classifier.

    This function serves as the entry point for the training script, utilizing Hydra 
    for configuration management. It handles reproducibility seeding, data loading, 
    model instantiation with dynamic parameters, optimizer/scheduler setup, 
    and integration with Weights & Biases for experiment tracking.

    Args:
        cfg (DictConfig): A hierarchical configuration object managed by Hydra 
            containing model architecture, training hyperparameters, and device settings.

    Returns:
        None
    """
   
    
    # Set random seed for reproducability 
    set_seed()
    
    # Get the project root, ensuring paths are absolute
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, "data")
    dataset_path = os.path.join(data_dir, "fcakyon___pokemon-classification")
    clean_dataset_path = os.path.join(data_dir, "pokemon_clean")
    
    # Create data/ dir and download/clean required data 
    create_data_dir(data_dir=data_dir, dataset_path=dataset_path, clean_dataset_path=clean_dataset_path)
    
    # Change run name as needed
    RUN_NAME = "pokemon_final_training_v2_50epochs"
    
   # 1. Accessing config values
    print(f"Training on: {cfg.device}") 
    
    # 2. Setup data (Accessing nested values)
    train_dl, val_dl, _ = create_dataloaders(
        clean_data_path=clean_dataset_path,
        batch_size=cfg.training.batch_size
    )

    # 3. Initialize model
    model = DynamicCNN(
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
            )

    # 4 .Init W&B using the Hydra config
    wandb_run = init_wandb_run(config=cfg, run_name=RUN_NAME)

    # 5 .Start training
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
  