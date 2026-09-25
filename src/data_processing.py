# data_processing.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


def load_and_preprocess_data(
    filepath: str, alphabet: str, start_token: str = "-", end_token: str = "."
) -> List[str]:
    """
    Load text from a file and preprocess them into whole names with specified start and end tokens.

    This function reads a file where each line contains a word followed by additional data
    (typically two numbers), all separated by spaces. It processes each word by adding specified
    start and end tokens, then stores each word as a sequence.

    Note: It is expected that each line in the file contains a word followed by two numerical
    data elements, all separated by spaces. The last two elements shall be ignored. Characters
    not found in the alphabet shall be ignored.

    Args:
        filepath (str): Path to the text file.
        alphabet (str): A string containing all unique characters to predict.
        start_token (str): A character used as the start token for each word.
        end_token (str): A character used as the end token for each word.

    Returns:
        List[str]: A list of whole names, where each name starts with a start token and ends with an end token.
    """
    with open(filepath, "r") as file:
        lines: List[str] = file.read().splitlines()

    names: List[str] = []
    for line in lines:
        # Splitting the line by spaces and discarding the last two elements
        parts: List[str] = line.strip().split()
        if len(parts) < 3:
            continue  # Skip lines that don't have enough parts
        word: str = " ".join(parts[:-2]).lower()  # Joining all parts except the last two

        # Filtering out characters not in the alphabet
        word = "".join([char for char in word if char in alphabet])

        # Adding start and end tokens to the word
        name = start_token + word + end_token
        names.append(name)

    return names


class CharTokenizer:
    """
    Character-level tokenizer for encoding and decoding text sequences.

    Args:
        alphabet (str): A string containing all unique characters in the dataset.
    """

    def __init__(self, alphabet: str, start_token: str = "-", end_token: str = "."):
        self.alphabet = sorted(list(set(alphabet)))
        self.char2idx = {char: idx + 3 for idx, char in enumerate(self.alphabet)}
        self.char2idx[start_token] = 1
        self.char2idx[end_token] = 2
        self.idx2char = {idx + 3: char for idx, char in enumerate(self.alphabet)}
        self.idx2char[1] = start_token
        self.idx2char[2] = end_token
        self.vocab_size = len(self.alphabet) + 3

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integer token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of integer token IDs.
        """
        return [self.char2idx[char] for char in text]

    def decode(self, indices: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a string.

        Args:
            indices (List[int]): A list of integer token IDs.

        Returns:
            str: The decoded string.
        """
        return ''.join([self.idx2char[idx] for idx in indices])


class NameDataset(Dataset):
    """
    PyTorch Dataset for loading encoded names.

    Args:
        encoded_names (List[List[int]]): A list of encoded names, where each name is a list of integer token IDs.
    """

    def __init__(self, encoded_names: List[List[int]]):
        self.encoded_names = encoded_names

    def __len__(self) -> int:
        return len(self.encoded_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and target sequences for a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing input_ids and target_ids tensors.
        """
        encoded_name = self.encoded_names[idx]
        input_ids = torch.tensor(encoded_name[:-1], dtype=torch.long)
        target_ids = torch.tensor(encoded_name[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences in the batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing input_ids and target_ids.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded input_ids and target_ids tensors.
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # Pad sequences to the maximum length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(
        target_ids, batch_first=True, padding_value=0
    )

    return input_ids_padded, target_ids_padded