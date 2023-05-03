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
    def f_postprocess(A):
        def fcompute(x, y, c):
            # suqeeze

            #clamp

            # rgb switch

            #transpose


            
        return te.compute((2560, 1792, 3), fcompute, name="postprocess")


    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 3, 2560, 1792], "float32"))
    with bb.function("postprocess", [x]):
        image = bb.emit(
            bb.call_te(f_postprocess, x, primfunc_name_hint="tir_postprocess")
        )
        bb.emit_func_output(image)
    return bb.get()

