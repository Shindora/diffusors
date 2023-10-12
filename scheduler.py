from diffusers import  DDPMScheduler
class CustomDDPMScheduler(DDPMScheduler):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.to(device)

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)

