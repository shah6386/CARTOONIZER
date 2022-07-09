import numpy as np
from PIL import Image

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def cartoonify(image, device=None):
    if torch.cuda.is_available() and (device is None or device == "gpu"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Generator()
    net.load_state_dict(torch.load('./weights/face_paint_512_v2.pt', map_location=device))
    net.to(device).eval()

    image = Image.fromarray(image).convert('RGB')
    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        if device is "gpu":
            out = net(image.to(device), True).cuda()
        else:
            out = net(image.to(device), True).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    return np.array(out)
