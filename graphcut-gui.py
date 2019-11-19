import argparse
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.draw import line, circle
import maxflow

import graphcut


class Slider:
    def __init__(self, master, label, *args, **kwargs):
        self.frame = tk.Frame(master)

        self.label = tk.Label(self.frame, text=label)
        self.slider = tk.Scale(self.frame, *args, **kwargs)

    def pack(self, *args, **kwargs):
        self.frame.pack(*args, **kwargs)
        self.label.pack(side=tk.LEFT)
        self.slider.pack(side=tk.LEFT)

    def bind(self, *args, **kwargs):
        self.slider.bind(*args, **kwargs)

    def get(self):
        return self.slider.get()


class GraphcutGUI(object):
    FORE = 1
    BACK = 2
    COLORS = {
        FORE: (255, 0, 0),
        BACK: (0, 0, 255)
    }

    def __init__(self, im):
        self.OGimage = np.array(Image.fromarray(im))
        self.image = self.OGimage.copy()
        self.height, self.width = self.image.shape[:2]
        self.anno = np.zeros((self.height, self.width))

        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack(side=tk.TOP)
        self.canvas.bind("<Button-1>", self.mark_image)
        self.canvas.bind("<B1-Motion>", self.mark_image)

        self.bottom = tk.Frame(self.root)
        self.bottom.pack(side=tk.BOTTOM)

        self.button_frame = tk.Frame(self.bottom)
        self.button_frame.pack(side=tk.TOP)

        self.fore_btn = tk.Button(self.button_frame, text="Fore", command=self.fore)
        self.fore_btn.pack(side=tk.LEFT)

        self.back_btn = tk.Button(self.button_frame, text="Back", command=self.back)
        self.back_btn.pack(side=tk.LEFT)

        self.seg_btn = tk.Button(self.button_frame, text="Seg", command=self.segment)
        self.seg_btn.pack(side=tk.LEFT)

        self.reset_btn = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(side=tk.LEFT)

        self.slider_frame = tk.Frame(self.bottom)
        self.slider_frame.pack(side=tk.TOP)

        self.lam_scale = Slider(self.slider_frame, "Lambda:", from_=0, to=1, orient=tk.HORIZONTAL, resolution=-1)
        self.lam_scale.pack(side=tk.TOP)

        self.sig_scale = Slider(self.slider_frame, "Sigma:", from_=0.001, to=1, orient=tk.HORIZONTAL, resolution=-1)
        self.sig_scale.pack(side=tk.TOP)

        self.display_image()

        self.lastX = 0
        self.lastY = 0
        self.ground = self.FORE

    def run(self):
        self.root.mainloop()

    def display_image(self):
        self.image[self.anno == self.FORE] = self.COLORS[self.FORE]
        self.image[self.anno == self.BACK] = self.COLORS[self.BACK]
        self.photoImage = ImageTk.PhotoImage(Image.fromarray(self.image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photoImage)

    def mark_image(self, event):
        x = min(max(event.x, 0), self.width-1)
        y = min(max(event.y, 0), self.height-1)

        rr, cc = circle(y, x, 5)
        rr = np.maximum(np.minimum(rr, self.height-1), 0)
        cc = np.maximum(np.minimum(cc, self.width-1), 0)

        self.anno[rr, cc] = self.ground
        self.display_image()

    def reset(self):
        self.image = self.OGimage.copy()
        self.height, self.width = self.image.shape[:2]
        self.anno = np.zeros((self.height, self.width))
        self.display_image()

    def fore(self):
        self.ground = self.FORE

    def back(self):
        self.ground = self.BACK

    def segment(self):
        lam = self.lam_scale.get()
        sig = self.sig_scale.get()
        fore = self.anno == 1
        back = self.anno == 2
        seg = graphcut.graphcut(self.OGimage, fore, back, lam, sig)
        self.image = self.OGimage.copy()
        self.image[seg == 0] = 0
        self.display_image()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imname")
    args = parser.parse_args()
    im = np.array(Image.open(args.imname).resize((224, 224)))

    GraphcutGUI(im).run()
