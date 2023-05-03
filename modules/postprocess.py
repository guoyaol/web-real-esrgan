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
    def f_swapchannel(A):
        def fcompute(c, x, y):
            return A[2-c, x, y]
        return te.compute((3, 2560, 1792), fcompute, name="swapnnel")
    
    def f_transpose(A):
        def fcompute(x, y, c):
            return A[c, y, x]
        return te.compute((2560, 1792, 3), fcompute, name="transpose")


    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 3, 2560, 1792], "float32"))
    with bb.function("postprocess", [x]):
        #squeeze
        squeezed = bb.emit(relax.op.squeeze(x))
        #clamp
        maxed = bb.emit(relax.op.maximum(squeezed, relax.op.const(0, "float32")))
        clamped = bb.emit(relax.op.minimum(maxed, relax.op.const(1, "float32")))
        #rgb swap
        swapped = bb.emit(bb.call_te(f_swapchannel, clamped, primfunc_name_hint="tir_swapchannel"))
        #transpose
        out_image = bb.emit(bb.call_te(f_transpose, swapped, primfunc_name_hint="tir_transpose"))

        bb.emit_func_output(out_image)
    return bb.get()

