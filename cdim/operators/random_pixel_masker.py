import torch

from cdim.operators import register_operator

@register_operator(name='random_inpainting')
class RandomPixelMasker:
    def __init__(self, height=256, width=256, channels=3, fraction=0.08, device='cpu'):
        """
        Initialize the ConsistentRandomPixelSelector.
        
        Args:
        height (int): Height of the input tensors (default: 256)
        width (int): Width of the input tensors (default: 256)
        channels (int): Number of channels in the input tensors (default: 3)
        fraction (float): Fraction of pixels to keep (default: 0.08 for 8%)
        device (str): Device to create the mask on (default: 'cpu')
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.fraction = fraction
        self.device = device
        
        # Create a binary mask for pixel selection
        num_pixels = height * width
        num_selected = int(num_pixels * fraction)
        self.mask = torch.zeros((1, channels, height, width), device=device)
        
        # Randomly select pixel indices
        selected_indices = torch.randperm(num_pixels)[:num_selected]
        
        # Convert indices to 2D coordinates
        selected_y = selected_indices // width
        selected_x = selected_indices % width
        
        # Set selected pixels in the mask to 1
        self.mask[0, :, selected_y, selected_x] = 1
    
    def __call__(self, tensor):
        """
        Apply the consistent random pixel selection to the input tensor.
        
        Args:
        tensor (torch.Tensor): Input tensor of shape (b, channels, height, width)
        
        Returns:
        torch.Tensor: Tensor with the same shape as input, but with only selected pixels
        """
        b, c, h, w = tensor.shape
        assert c == self.channels and h == self.height and w == self.width, \
            f"Input tensor must be of shape (b, {self.channels}, {self.height}, {self.width})"
        
        # Move the mask to the same device as the input tensor if necessary
        if tensor.device != self.mask.device:
            self.mask = self.mask.to(tensor.device)
        
        # Apply the mask to the input tensor
        return tensor * self.mask
