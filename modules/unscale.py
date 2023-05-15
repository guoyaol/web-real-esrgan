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

def unscale_image() -> tvm.IRModule:
    from tvm import te
    #divide each element by 255
    #todo: different sizes of images
    def f_unscale_image(A):
        def fcompute(y, x, c):
            return te.round(A[y, x, c] * 255).astype("uint8")

        return te.compute((716, 716, 3), fcompute, name="unscale_image")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([716, 716, 3], "float32"))
    with bb.function("unscale_image", [x]):
        image = bb.emit(
            bb.call_te(f_unscale_image, x, primfunc_name_hint="tir_unscale_image")
        )
        bb.emit_func_output(image)
    return bb.get()


img = torch.rand((716, 716, 3), dtype=torch.float32)

img = img.numpy()


# our result

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()

img_nd = tvm.nd.array(img.astype("float32"), device=device)
u_mod = unscale_image()

from tvm import meta_schedule as ms
db = ms.database.create(work_dir="scale_db")
with target, db, tvm.transform.PassContext(opt_level=3):
    u_mod = relax.transform.MetaScheduleApplyDatabase()(u_mod)
    u_mod = tvm.tir.transform.DefaultGPUSchedule()(u_mod)

ex = relax.build(u_mod, target= target)
vm = relax.VirtualMachine(ex, device)

nd_res1 = vm["unscale_image"](img_nd)


# ref result
img = img.astype(np.float32)
max_range = 255
img = (img * 255.0).round().astype(np.uint8)
print(img)

np.testing.assert_array_equal(nd_res1.numpy(), img)
print("test passed")