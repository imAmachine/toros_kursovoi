import torch
from torch.utils.data import Dataset

def add_noise_to_mask(mask, noise_type='gaussian', noise_level=0.05):
    """
    Добавление шума к бинарной маске
    
    Args:
        mask (torch.Tensor): Входная бинарная маска
        noise_type (str): Тип шума ('gaussian', 'salt_and_pepper')
        noise_level (float): Интенсивность шума
    
    Returns:
        torch.Tensor: Маска с добавленным шумом
    """
    mask = mask.clone()
    
    if noise_type == 'gaussian':
        # Гауссов шум
        noise = torch.randn_like(mask) * noise_level
        noisy_mask = mask + noise
        noisy_mask = torch.clamp(noisy_mask, 0, 1)
    
    elif noise_type == 'salt_and_pepper':
        # Импульсный шум (соль и перец)
        noise_mask = torch.rand_like(mask)
        
        # Создаем маску для "соли" (белые пиксели)
        salt_mask = noise_mask < noise_level / 2
        # Создаем маску для "перца" (черные пиксели)
        pepper_mask = noise_mask > 1 - noise_level / 2
        
        noisy_mask = mask.clone()
        noisy_mask[salt_mask] = 1.0
        noisy_mask[pepper_mask] = 0.0
    
    else:
        raise ValueError(f"Неподдерживаемый тип шума: {noise_type}")
    
    return noisy_mask

class IceRidgeDataset(Dataset):
    def __init__(self, examples, transform=None, noise_type='gaussian', noise_level=0.05):
        self.examples = examples
        self.transform = transform
        self.noise_type = noise_type
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_mask, target_mask = self.examples[idx]
        
        # Преобразование в тензоры
        input_mask = torch.from_numpy(input_mask).float().unsqueeze(0)
        target_mask = torch.from_numpy(target_mask).float().unsqueeze(0)
        
        # Добавление шума
        input_mask = add_noise_to_mask(
            input_mask, 
            noise_type=self.noise_type, 
            noise_level=self.noise_level
        )
        
        # Применение трансформаций
        if self.transform:
            input_mask = self.transform(input_mask)
            target_mask = self.transform(target_mask)
        
        return input_mask, target_mask