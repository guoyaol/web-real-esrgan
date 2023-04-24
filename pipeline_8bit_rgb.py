import numpy as np
import cv2
import os
import torch
import math
from torch.nn import functional as F
from network import RRDBNet
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
import torch



input_path = "./input/OST_009.png"
output_path = "./output"

imgname, extension = os.path.splitext(os.path.basename(input_path))
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    z = torch.rand((1, 3, 640, 448), dtype=torch.float32)

    mod = dynamo_capture_subgraphs(
        rrdb.forward,
        z,
        keep_params_as_input=True,
    )
    assert len(mod.functions) == 1

    return tvm.IRModule({"rrdb": mod["subgraph_0"]})

def scale_image() -> tvm.IRModule:
    from tvm import te
    #divide each element by 255
    #todo: different sizes of images
    def f_scale_image(A):
        def fcompute(y, x, c):
            return A[y, x, c] / 255

        return te.compute((640, 448, 3), fcompute, name="scale_image")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([640, 448, 3], "float32"))
    with bb.function("scale_image", [x]):
        image = bb.emit(
            bb.call_te(f_scale_image, x, primfunc_name_hint="tir_scale_image")
        )
        bb.emit_func_output(image)
    return bb.get()

def preprocess() -> tvm.IRModule:
    from tvm import te
    #np.transpose(img, (2, 0, 1)) and unqueeze(0)
    #todo: different sizes of images
    def f_preprocess(A):
        def fcompute(i, c, x, y):
            return A[x, y, c]
        return te.compute((1, 3, 640, 448), fcompute, name="preprocess")


    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([640, 448, 3], "float32"))
    with bb.function("preprocess", [x]):
        image = bb.emit(
            bb.call_te(f_preprocess, x, primfunc_name_hint="tir_preprocess")
        )
        bb.emit_func_output(image)
    return bb.get()

#1. scale image
img = img.astype(np.float32)
max_range = 255
img = img / max_range


#2. preprocess image
img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
img = img.unsqueeze(0)


# 3. model inference
loadnet = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(loadnet['params_ema'], strict=True)



mod = rrdb_net(model)
mod, params = relax.frontend.detach_params(mod)
mod = relax.transform.LegalizeOps()(mod)

ex = relax.build(mod, target= "llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

img = tvm.nd.array(img)
nd_res = vm["rrdb"](img, params)
print(nd_res)


# #4. post process
# output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
# output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))


# #5. re-scale image

# output = (output_img * 255.0).round().astype(np.uint8)





# #---------------------save image---------------------
# extension = extension[1:]

# save_path = os.path.join(output_path, f'{imgname}.{extension}')

# cv2.imwrite(save_path, output)