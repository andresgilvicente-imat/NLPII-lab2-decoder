# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import CharTokenizer
from torch.utils.data import DataLoader
from data_processing import load_and_preprocess_data, CharTokenizer, NameDataset, collate_fn
from decoder import TransformerForLanguageModeling  # Import your model class

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    num_epochs: int,
    learning_rate: float,
    model_save_dir: str,
    model_params: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Train a Transformer decoder model for name generation.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        tokenizer (CharTokenizer): The tokenizer used for encoding/decoding.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        model_save_dir (str): Directory to save the trained model.
        model_params (dict): Dictionary of model hyperparameters.
        device (torch.device): Device to run the training on.
    """

    # Ensure the model save directory exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Initialize the model
    print("Initializing model...")
    vocab_size = tokenizer.vocab_size
    model = TransformerForLanguageModeling(
        vocab_size=vocab_size,
        **model_params
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits = model(input_ids)
            logits = logits.view(-1, vocab_size)
            targets = target_ids.view(-1)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                logits = logits.view(-1, vocab_size)
                targets = target_ids.view(-1)

                loss = criterion(logits, targets)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")

    return model

# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split
    from data_processing import load_and_preprocess_data, CharTokenizer, NameDataset, collate_fn

    # Define parameters
    data_filepath = "data/nombres_raw.txt"  # Replace with your actual file path
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    start_token = "-"
    end_token = "."
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-4
    model_save_dir = "runs"

    model_params = {
        "d_model": 8,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "max_position_embeddings": len(alphabet) + 2,  # Set as needed
    }

    # Load and preprocess data
    print("Loading and preprocessing data...")
    names = load_and_preprocess_data(data_filepath, alphabet, start_token, end_token)

    # Initialize tokenizer
    tokenizer = CharTokenizer(alphabet + start_token + end_token)

    # Encode names
    print("Encoding names...")
    encoded_names = [tokenizer.encode(name) for name in names]

    # Create dataset
    dataset = NameDataset(encoded_names)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Call the train function
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model_save_dir=model_save_dir,
        model_params=model_params,
        device=torch.device("cpu"),
    )
