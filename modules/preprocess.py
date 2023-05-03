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

input_path = "/home/guoyaol/web-real-esrgan/input/OST_009.png"

imgname, extension = os.path.splitext(os.path.basename(input_path))
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img = img.astype(np.float32)
max_range = 255
img = img / max_range

#our result
print("our result")
img_nd = tvm.nd.array(img.reshape(640, 448, 3).astype("float32"))

p_mod = preprocess()

ex = relax.build(p_mod, target= "llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
nd_res1 = vm["preprocess"](img_nd)

print(nd_res1)
print(nd_res1.shape)


#ref result
print("ref result")
img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
img = img.unsqueeze(0)
print(img)
print(img.shape)

np.testing.assert_array_equal(nd_res1.numpy(), img.numpy())
print("test passed")

