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


def postprocess() -> tvm.IRModule:
    from tvm import te
    # output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    def f_squeeze(A):
        def fcompute(c, x, y):
            return A[0, c, x, y]
        return te.compute((3, 716, 716), fcompute, name="squeeze")

    def f_swapchannel(A):
        def fcompute(c, x, y):
            return A[2-c, x, y]
        return te.compute((3, 716, 716), fcompute, name="swapnnel")
    
    def f_transpose(A):
        def fcompute(x, y, c):
            return A[c, x, y]
        return te.compute((716, 716, 3), fcompute, name="transpose")
    
    def f_max_0(A):
        def fcompute(c, x, y):
            return te.if_then_else(A[c, x, y] > te.const(0, "float32"), A[c, x, y], te.const(0, "float32"))
        return te.compute((3, 716, 716), fcompute, name="max0")
    
    def f_min_1(A):
        def fcompute(c, x, y):
            return te.if_then_else(A[c, x, y] < te.const(1, "float32"), A[c, x, y], te.const(1, "float32"))
        return te.compute((3, 716, 716), fcompute, name="min1")


    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 3, 716, 716], "float32"))
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


img = torch.rand((1, 3, 716, 716), dtype=torch.float32)

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()

img_nd = tvm.nd.array(img, device=device)


#our result
print("our result")


from tvm import meta_schedule as ms
db = ms.database.create(work_dir="scale_db")

p_mod = postprocess()
with target, db, tvm.transform.PassContext(opt_level=3):
    p_mod = relax.transform.MetaScheduleApplyDatabase()(p_mod)
    p_mod = tvm.tir.transform.DefaultGPUSchedule()(p_mod)
ex = relax.build(p_mod, target= target)
vm = relax.VirtualMachine(ex, device)

nd_res1 = vm["postprocess"](img_nd).numpy()

print(nd_res1)
print(nd_res1.shape)


#ref result
print("ref result")
output_img = img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

print(output_img)
print(output_img.shape)

np.testing.assert_array_equal(nd_res1, output_img)
print("test passed")
