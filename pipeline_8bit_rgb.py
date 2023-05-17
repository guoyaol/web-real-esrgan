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

def unscale_image() -> tvm.IRModule:
    from tvm import te
    #divide each element by 255
    #todo: different sizes of images
    def f_unscale_image(A):
        def fcompute(y, x, c):
            return te.round(A[y, x, c] * 255).astype("uint8")

        return te.compute((2560, 1792, 3), fcompute, name="unscale_image")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([2560, 1792, 3], "float32"))
    with bb.function("unscale_image", [x]):
        image = bb.emit(
            bb.call_te(f_unscale_image, x, primfunc_name_hint="tir_unscale_image")
        )
        bb.emit_func_output(image)
    return bb.get()

#1. scale image
scale = scale_image()

#2. preprocess image
pre_pro = preprocess()

# 3. model inference
loadnet = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(loadnet['params_ema'], strict=True)
rrdb = rrdb_net(model)

#4. post process
post_pro = postprocess()

#5. un-scale image
unscale = unscale_image()

#---------------------merge together---------------------
def merge_irmodules(*irmodules: tvm.IRModule) -> tvm.IRModule:
    merged_mod = tvm.IRModule()

    for mod in irmodules:
        for gv, func in mod.functions.items():
            merged_mod[gv] = func
    return merged_mod

mod: tvm.IRModule = merge_irmodules(
    scale,
    pre_pro,
    rrdb,
    post_pro,
    unscale
)

mod, params = relax.frontend.detach_params(mod)

mod = relax.pipeline.get_pipeline()(mod)

entry_funcs = ["scale_image", "preprocess", "rrdb", "postprocess", "unscale_image"]
mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)

mod = relax.transform.LiftTransformParams()(mod)

for global_var, function in mod.functions.items():
    if isinstance(function, relax.Function):
        if global_var.name_hint.endswith("_transform_params"):
            print(
                global_var.name_hint,
                f' # <=== This is the weight parameter computation function for "{global_var.name_hint[:-17]}"',
            )
        else:
            print(global_var.name_hint)

def print_relax_funcnames(mod: tvm.IRModule):
    for global_var, func in mod.functions.items():
        if isinstance(func, relax.Function):
            print(global_var.name_hint)
    print()

def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str], mod_deploy_entry_func: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_names = [name + "_transform_params" for name in model_names]
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint in transform_func_names:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination(transform_func_names)(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(mod_deploy_entry_func)(
        mod_deploy
    )

    return mod_transform, mod_deploy

model_names = ["rrdb"]

mod_transform, mod_deploy = split_transform_deploy_mod(
    mod, model_names, entry_funcs
)

print("In IRModule for build stage:")
print_relax_funcnames(mod_transform)

print("In IRModule for deployment stage:")
print_relax_funcnames(mod_deploy)


def transform_params(
    mod_transform: tvm.IRModule, model_params: Dict[str, List[tvm.nd.NDArray]]
) -> Dict[str, List[tvm.nd.NDArray]]:
    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())
    new_params = dict()
    for name, params in model_params.items():
        new_params[name] = vm[name + "_transform_params"](params)
    return new_params


def save_params(params: Dict[str, List[tvm.nd.NDArray]], artifact_path: str) -> None:
    from tvm.contrib import tvmjs

    meta_data = {}
    param_dict = {}
    for model in ["rrdb"]:
        meta_data[f"{model}ParamSize"] = len(params[model])
        for i, nd in enumerate(params[model]):
            param_dict[f"{model}_{i}"] = nd
    tvmjs.dump_ndarray_cache(param_dict, f"{artifact_path}/params", meta_data=meta_data)

new_params = transform_params(mod_transform, params)
save_params(new_params, artifact_path="dist")

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()

def tune(mod: tvm.IRModule) -> None:
    from tvm import meta_schedule as ms

    ms.relax_integration.tune_relax(
        mod=mod,
        target=tvm.target.Target("apple/m1-gpu-restricted"),
        params={},
        builder=ms.builder.LocalBuilder(
            max_workers=6,
        ),
        runner=ms.runner.RPCRunner(
            ms.runner.RPCConfig(
                tracker_host="192.168.10.1",
                tracker_port=9191,
                tracker_key="m2-mac-mini",
                session_timeout_sec=50,
            )
        ),
        work_dir="log_db_tuning",
        max_trials_global=100,
        max_trials_per_task=4,
    )

tune(mod_deploy)



# from tvm import meta_schedule as ms

# db = ms.database.create(work_dir="log_db")
# with target, db, tvm.transform.PassContext(opt_level=3):
#     mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
#     mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)

# ex = relax.build(mod=mod_deploy, target=target)

# ex.export_library("dist/real_esrgan.so")

# target = tvm.target.Target("apple/m2-gpu")
# device = tvm.metal()

# def load_params(artifact_path: str, device) -> Dict[str, List[tvm.nd.NDArray]]:
#     from tvm.contrib import tvmjs

#     pdict = {}
#     params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
#     for model in ["rrdb"]:
#         plist = []
#         size = meta[f"{model}ParamSize"]
#         for i in range(size):
#             plist.append(params[f"{model}_{i}"])
#         pdict[model] = plist
#     return pdict

# const_params_dict = load_params(artifact_path="dist", device=device)
# ex = tvm.runtime.load_module("dist/real_esrgan.so")

# vm = relax.VirtualMachine(rt_mod=ex, device=device)

# class TVMESRPipeline:
#     def __init__(
#         self,
#         vm: relax.VirtualMachine,
#         tvm_device,
#         param_dict,
#     ):
#         def wrapper(f, params):
#             def wrapped_f(*args):
#                 return f(*args, params)

#             return wrapped_f

#         self.vm = vm
        
#         self.rrdb = wrapper(vm["rrdb"], param_dict["rrdb"])
        
#         self.scale_image = vm["scale_image"]
#         self.preprocess = vm["preprocess"]
#         self.unscale_image = vm["unscale_image"]
#         self.postprocess = vm["postprocess"]

#         self.tvm_device = tvm_device
#         self.param_dict = param_dict

#     def __call__(self, input_image: np.array):
#         image = tvm.nd.array(input_image, device=self.tvm_device)
#         image = self.scale_image(image)
#         image = self.preprocess(image)
#         image = self.rrdb(image)
#         image = self.postprocess(image)
#         image = self.unscale_image(image)

#         return image
    
# pipe = TVMESRPipeline(vm, device, const_params_dict)
