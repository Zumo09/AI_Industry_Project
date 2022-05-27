import torch

def load_model(
    base_model: torch.nn.Module, load_path: str, device: torch.device
) -> torch.nn.Module:
    state_dict = torch.load(load_path, map_location=device)
    base_model.load_state_dict(state_dict)
    return base_model.to(device)


def save_model(model: torch.nn.Module, save_path: str) -> None:
    torch.save(model.state_dict(), save_path)