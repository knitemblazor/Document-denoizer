import cv2
from models import *
from PIL import Image
from torchvision.utils import *
import torchvision.transforms as transforms


class DeNoiser:

    def __init__(self):
        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            self.generator = GeneratorUNet().cuda()
            self.generator.load_state_dict(torch.load("saved_models/generator_alter.pth"))
        else:
            self.generator = GeneratorUNet()
            self.generator.load_state_dict(torch.load("saved_models/generator_alter.pth", map_location=torch.device('cpu')))

    def den(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(gray)
        transform = transforms.Compose([ transforms.ToTensor(), ])
        img_A = transform(img)
        if self.cuda:
            img_A = img_A.reshape(1, 3, 256, 256).cuda()
        else:
            img_A = img_A.reshape(1, 3, 256, 256)
        hr = self.generator(img_A)
        grid = make_grid(hr)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return ndarr
