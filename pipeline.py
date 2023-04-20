import numpy as np
import cv2
import os
import torch
import math
from torch.nn import functional as F
from network import RRDBNet



netscale = 4
model_path = "./weights/RealESRGAN_x4plus.pth"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
outscale = 4




input_path = "./input/children-alpha.png"
output_path = "./output"

imgname, extension = os.path.splitext(os.path.basename(input_path))

img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

if len(img.shape) == 3 and img.shape[2] == 4:
    img_mode = 'RGBA'
else:
    img_mode = None


#1. load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loadnet = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(loadnet['params_ema'], strict=True)
model.to(device)
model.eval()

with torch.no_grad():
    alpha_upsampler='realesrgan'

    #2. scale image
    h_input, w_input = img.shape[0:2]
    img = img.astype(np.float32)
    if np.max(img) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
    img = img / max_range

    #3. convert image to RGB, color spcace
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if alpha_upsampler == 'realesrgan':
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #4. preprocess image
    def preprocess(img, device):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        return img

    img = preprocess(img, device)

    #5. model inference
    output_img = model(img)


    #6. post process
    output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    if img_mode == 'L':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    #7. process the alpha channel if necessary ------------------- #
    if img_mode == 'RGBA':
        if alpha_upsampler == 'realesrgan':
            alpha = preprocess(alpha, device)

            output_alpha = model(alpha)

            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    #8. un-scale image
    if max_range == 65535:  # 16-bit image
        output = (output_img * 65535.0).round().astype(np.uint16)
    else:
        output = (output_img * 255.0).round().astype(np.uint8)


    #9. save the output image
    extension = extension[1:]

    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = 'png'

    save_path = os.path.join(output_path, f'{imgname}.{extension}')

    cv2.imwrite(save_path, output)