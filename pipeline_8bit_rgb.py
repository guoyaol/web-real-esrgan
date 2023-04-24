import numpy as np
import cv2
import os
import torch
import math
from torch.nn import functional as F
from network import RRDBNet
import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
import torch



input_path = "./input/OST_009.png"
output_path = "./output"

imgname, extension = os.path.splitext(os.path.basename(input_path))
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)


netscale = 4
model_path = "./weights/RealESRGAN_x4plus.pth"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
outscale = 4

def rrdb_net(model) -> tvm.IRModule:

    class RRDBNetWrapper(torch.nn.Module):
        def __init__(self, rrdb):
            super().__init__()
            self.rrdb = rrdb

        def forward(self, input):
            output = self.rrdb(input)
            return output

    rrdb = RRDBNetWrapper(model)

    #todo: change size
    z = torch.rand((1, 3, 128, 128), dtype=torch.float32)

    mod = dynamo_capture_subgraphs(
        rrdb.forward,
        z,
        keep_params_as_input=True,
    )
    assert len(mod.functions) == 1

    return tvm.IRModule({"rrdb": mod["subgraph_0"]})



#2. scale image
h_input, w_input = img.shape[0:2]
img = img.astype(np.float32)

max_range = 255

img = img / max_range

#3. convert image to RGB, color spcace
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#4. preprocess image
#1. load model
def preprocess(img):
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.unsqueeze(0)
    return img

img = preprocess(img)



loadnet = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(loadnet['params_ema'], strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mod = rrdb_net(model)
mod, params = relax.frontend.detach_params(mod)
# mod.show()
model.to(device)
model.eval()

with torch.no_grad():
#5. model inference
    output_img = model(img.to(device))

# import GPUtil
# has_gpu = len(GPUtil.getGPUs()) > 0


# target = tvm.target.Target("cuda" if has_gpu else "llvm")
# device = tvm.cuda() if has_gpu else tvm.cpu()

# ex = relax.build(mod, target= "llvm")
# vm = relax.VirtualMachine(ex, device)
# nd_res = vm["rrdb"](img, params)
# print(nd_res)


#6. post process
output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))


#8. un-scale image

output = (output_img * 255.0).round().astype(np.uint8)





#---------------------save image---------------------
extension = extension[1:]

save_path = os.path.join(output_path, f'{imgname}.{extension}')

cv2.imwrite(save_path, output)