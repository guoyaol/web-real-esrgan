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
from typing import Dict, List, Tuple
import time



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
        def fcompute(x, y, c):
            return A[x, y, c] / te.const(255, "float32")

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

def unscale_image() -> tvm.IRModule:
    from tvm import te
    #divide each element by 255
    #todo: different sizes of images
    def f_unscale_image(A):
        def fcompute(y, x, c):
            return A[y, x, c] * 255

        return te.compute((640, 448, 3), fcompute, name="unscale_image")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([640, 448, 3], "float32"))
    with bb.function("unscale_image", [x]):
        image = bb.emit(
            bb.call_te(f_unscale_image, x, primfunc_name_hint="tir_unscale_image")
        )
        bb.emit_func_output(image)
    return bb.get()

def postprocess() -> tvm.IRModule:
    from tvm import te
    # output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    def f_squeeze(A):
        def fcompute(c, x, y):
            return A[0, c, x, y]
        return te.compute((3, 2560, 1792), fcompute, name="squeeze")

    def f_swapchannel(A):
        def fcompute(c, x, y):
            return A[2-c, x, y]
        return te.compute((3, 2560, 1792), fcompute, name="swapnnel")
    
    def f_transpose(A):
        def fcompute(x, y, c):
            return A[c, x, y]
        return te.compute((2560, 1792, 3), fcompute, name="transpose")
    
    def f_max_0(A):
        def fcompute(c, x, y):
            return te.if_then_else(A[c, x, y] > te.const(0, "float32"), A[c, x, y], te.const(0, "float32"))
        return te.compute((3, 2560, 1792), fcompute, name="max0")
    
    def f_min_1(A):
        def fcompute(c, x, y):
            return te.if_then_else(A[c, x, y] < te.const(1, "float32"), A[c, x, y], te.const(1, "float32"))
        return te.compute((3, 2560, 1792), fcompute, name="min1")


    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 3, 2560, 1792], "float32"))
    with bb.function("postprocess", [x]):
        #squeeze
        squeezed = bb.emit(bb.call_te(f_squeeze, x, primfunc_name_hint="tir_squeeze"))
        #clamp
        maxed = bb.emit(bb.call_te(f_max_0, squeezed, primfunc_name_hint="tir_max_0"))
        clamped = bb.emit(bb.call_te(f_min_1, maxed, primfunc_name_hint="tir_min_1"))
        #rgb swap
        swapped = bb.emit(bb.call_te(f_swapchannel, clamped, primfunc_name_hint="tir_swapchannel"))
        #transpose
        out_image = bb.emit(bb.call_te(f_transpose, swapped, primfunc_name_hint="tir_transpose"))

        bb.emit_func_output(out_image)
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
# ex = relax.build(mod, target= "cuda")
# vm = relax.VirtualMachine(ex, tvm.cuda())

img = tvm.nd.array(img)

#record inference time
start = time.time()

print("start inference")
nd_res = vm["rrdb"](img, *params['rrdb'])
print(nd_res)

end = time.time()
print("inference time in seconds: ", end - start)


# #4. post process
# output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
# output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))


# #5. re-scale image

# output = (output_img * 255.0).round().astype(np.uint8)





# #---------------------save image---------------------
# extension = extension[1:]

# save_path = os.path.join(output_path, f'{imgname}.{extension}')

# cv2.imwrite(save_path, output)