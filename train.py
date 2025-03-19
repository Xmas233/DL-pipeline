import os
import torch
import data_setup, engine, utils
from models.TinyVGG import TinyVGG
from torchvision import transforms
import swanlab

if __name__ == "__main__":
    # Setup hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    swanlab.init(
        project="Classification",

        config={
            "NUM_EPOCHS": NUM_EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "HIDDEN_UNITS": HIDDEN_UNITS,
            "LEARNING_RATE": LEARNING_RATE,
            "datasets": "food",
            "arcgitecture": "TinyVGG"
        }
    )

    # Setup paths
    TRAIN_DIR = "data/pizza_steak_sushi/train"
    TEST_DIR = "data/pizza_steak_sushi/train"

    # Setup target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=transform,
        batch_size=BATCH_SIZE
    )

    # Create model
    model = TinyVGG.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=NUM_EPOCHS)

    utils.save_model(model=model, target_dir="ckpt/final", model_name="tiny_vgg.pth")