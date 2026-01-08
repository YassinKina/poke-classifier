from src.data_setup import create_dataloaders

def train():
    #Get DatasetDict of pokemon pictures
    train_dl, val_dl, test_dl = create_dataloaders()
    # Grab the first batch from the train_loader
    images, labels = next(iter(train_dl))

    print(f"Batch Image Shape: {images.shape}") 
    # Expected: [batch_size, 3, 224, 224] -> e.g., [32, 3, 224, 224]

    print(f"Batch Label Shape: {labels.shape}")
    # Expected: [batch_size] -> e.g., [32]

    print(f"Label Data Type: {labels.dtype}")
    # Expected: torch.int64 (LongTensor)

    print(f"Image Pixel Range: Min={images.min():.2f}, Max={images.max():.2f}")
    # If normalized, you'll see negative numbers and values around 0-2.
    
   

    #use with custom dataset
train()