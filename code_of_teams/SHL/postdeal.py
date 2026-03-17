import glob
import os.path

import numpy as np
import cv2


def pix_meaning():


    img_dir = [
        "output_Uformer/",
        "output_restormer/",
        "output_mstpp/"
    ]
    weights = [0.333, 0.333, 0.333] 

    print(list(img_dir))
    print(weights)

    img_file = [sorted(glob.glob(f"{i}/*")) for i in img_dir]

    for fs in zip(*img_file):
        for fsp in range(1,len(fs)):
            assert(os.path.basename(fs[fsp-1]) == os.path.basename(fs[fsp]))

        img_all = [cv2.imread(i)[:,:,::-1] for i in fs]

        img_mean = sum([img_all[i] * weights[i] for i in range(len(img_all))])
        img_mean = np.clip(np.around(img_mean, 0), 0, 255).astype(np.uint8)

        if not os.path.exists('output_mean'):
            os.makedirs('output_mean', exist_ok=True)
        cv2.imwrite(os.path.join("output_mean/",str(os.path.basename(fs[0])).replace(".png",".jpg")), img_mean[:,:,::-1])


if __name__ == '__main__':
    pix_meaning()