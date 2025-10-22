import torch

from cdim.operators import register_operator

@register_operator(name='box_inpainting')
class RandomBoxMasker:
    def __init__(self, height=256, width=256, channels=3, box_size=128, device='cpu'):
        """
        Initialize the ConsistentBoxMasker with random box positioning.
        
        Args:
        height (int): Height of the input tensors (default: 256)
        width (int): Width of the input tensors (default: 256)
        channels (int): Number of channels in the input tensors (default: 3)
        box_size (int): Size of the box to mask (default: 128)
        device (str): Device to create the mask on (default: 'cpu')
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.box_size = min(box_size, height, width)  # Ensure box_size doesn't exceed image dimensions
        self.device = device
        
        # Create a binary mask for box selection
        self.mask = torch.ones((1, channels, height, width), device=device)
        
        # Randomly calculate the top-left corner of the box
        max_y = height - self.box_size
        max_x = width - self.box_size

        start_y = torch.randint(0, max_y + 1, (1,)).item()
        start_x = torch.randint(0, max_x + 1, (1,)).item()
        
        # Set the box area in the mask to 0
        self.mask[0, :, start_y:start_y+self.box_size, start_x:start_x+self.box_size] = 0
    
    def __call__(self, tensor):
        """
        Apply the consistent box masking to the input tensor.
        
        Args:
        tensor (torch.Tensor): Input tensor of shape (b, channels, height, width)
        
        Returns:
        torch.Tensor: Tensor with the same shape as input, but with the box area masked out
        """
        b, c, h, w = tensor.shape
        assert c == self.channels and h == self.height and w == self.width, \
            f"Input tensor must be of shape (b, {self.channels}, {self.height}, {self.width})"
        
        # Move the mask to the same device as the input tensor if necessary
        if tensor.device != self.mask.device:
            self.mask = self.mask.to(tensor.device)
        
        # Apply the mask to the input tensor
        return tensor * self.mask
    
    def select(self, tensor):
        """
        Extract only the observed pixels from the tensor (pixels outside the box).
        
        Args:
        tensor (torch.Tensor): Input tensor of shape (b, channels, height, width)
        
        Returns:
        torch.Tensor: Flattened tensor containing only observed pixels (b, num_observed)
        """
        b, c, h, w = tensor.shape
        assert c == self.channels and h == self.height and w == self.width, \
            f"Input tensor must be of shape (b, {self.channels}, {self.height}, {self.width})"
        
        # Move the mask to the same device as the input tensor if necessary
        if tensor.device != self.mask.device:
            self.mask = self.mask.to(tensor.device)
        
        # Extract only observed pixels (where mask is 1, outside the box)
        observed = (tensor * self.mask).flatten(1)  # (b, c*h*w)
        # Keep only non-zero elements
        mask_flat = self.mask.flatten(1)  # (1, c*h*w)
        return observed[:, mask_flat[0] > 0]  # (b, num_observed)
    
    def get_num_observed(self):
        """Return the number of observed elements."""
        return int(self.mask.sum().item())