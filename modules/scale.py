import numpy as np
import cv2
import os
import torch
import math
from torch.nn import functional as F
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
import torch

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


input_path = "/home/guoyaol/web-real-esrgan/input/OST_009.png"

imgname, extension = os.path.splitext(os.path.basename(input_path))
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# our result
img_nd = tvm.nd.array(img.astype("float32"))

s_mod = scale_image()

ex = relax.build(s_mod, target= "llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
nd_res1 = vm["scale_image"](img_nd)

print(nd_res1)
print(type(nd_res1))

# ref result
img = img.astype(np.float32)
max_range = 255
img = img / max_range
print(img)