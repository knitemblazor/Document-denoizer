import numpy as np
import cv2
from denoiser import DeNoiser
from joblib import parallel_backend
from joblib import Parallel, delayed
from PIL import Image


class ReSolver:
    def __init__(self, img):
        try:
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            self.gray = img
        self.y, self.x = self.gray.shape
        self.denoiser = DeNoiser()

    def nona4(self, img):
        canvas = np.array(Image.new('L', (2560, 3584), color=(256)))
        y, x = img.shape
        canvas[:y, :x] = img
        return canvas

    def executioner(self):
        if self.x < 2560 and self.y < 3584:
            self.flag = False
            resized = self.nona4(self.gray)
        else:
            self.flag = True
            resized = cv2.resize(self.gray, (2560, 3584), interpolation=cv2.INTER_CUBIC)
        tiles = []
        for i in range(14):
            m = i * 256
            for j in range(10):
                l = 256 * j
                tile = resized[m:m + 256, l:l + 256]
                tiles.append(tile)
        self.tiles = tiles

    def de_noiser(self, tile):
        tile = self.denoiser.den(tile)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        return tile

    def para(self):
        with parallel_backend('threading', n_jobs=4):
            self.tiles = Parallel()(delayed(self.de_noiser)(tile) for tile in self.tiles)

    def canvas(self):
        self.executioner()
        self.para()
        img = Image.new('L', (2560, 3584), color=(256))
        img = np.array(img)
        counter = 0
        for i in range(14):
            m = i * 256
            for j in range(10):
                l = 256 * j
                img[m:m + 256, l:l + 256] = self.tiles[counter]
                counter = counter + 1
        return img

    def orifice(self):
        img = self.canvas()
        if self.flag:
            pass
        else:
            img = img[:self.y, :self.x]
        return Image.fromarray(img)




