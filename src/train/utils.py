import torch


def get_device(device: str = "") -> torch.device:
    """
    Get the device to use for training.
    Returns:
        torch.device: The device to use for training.
    """
    if device != "":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps or torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Define a mapping from string to torch dtype
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
