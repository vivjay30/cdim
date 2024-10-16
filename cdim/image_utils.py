from torchvision.transforms import ToPILImage

def save_to_image(tensor, filename):
    """
    Saves a torch tensor to an image.
    The image assumed to be (1, 3, H, W)
    with values between (-1, 1)
    """
    to_save = (tensor[0] + 1) / 2
    to_save = to_save.clamp(0, 1)

    # Convert to PIL Image
    transform = ToPILImage()
    img = transform(to_save)

    # Save the image
    img.save(filename)
