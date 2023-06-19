from typing import Dict, List, Tuple

import os
import argparse
import pickle
import web_real_esrgan.trace as trace
import web_real_esrgan.utils as utils
from network import RRDBNet
from platform import system
import torch

import tvm
from tvm import relax


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str, default="auto")
    args.add_argument("--db-path", type=str, default="log_db_tuning_1000_small/")
    args.add_argument("--model-path", type=str, default="weights/RealESRGAN_x4plus.pth")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument("--debug-dump", action="store_true", default=False)

    parsed = args.parse_args()

    if parsed.target == "auto":
        if system() == "Darwin":
            target = tvm.target.Target("apple/m1-gpu")
        else:
            has_gpu = tvm.cuda().exist
            target = tvm.target.Target("cuda" if has_gpu else "llvm")
        print(f"Automatically configuring target: {target}")
        parsed.target = tvm.target.Target(target, host="llvm")
    elif parsed.target == "webgpu":
        parsed.target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )
    else:
        parsed.target = tvm.target.Target(parsed.target, host="llvm")
    return parsed


def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_dump_shader(ex, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def trace_models(
    args
) -> Tuple[tvm.IRModule, Dict[str, List[tvm.nd.NDArray]]]:

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=True)

    scale = trace.scale_image()

    #2. preprocess image
    pre_pro = trace.preprocess()

    # 3. model inference
    rrdb = trace.rrdb_net(model)

    #4. post process
    post_pro = trace.postprocess()

    #5. un-scale image
    unscale = trace.unscale_image()

    image_to_rgba = trace.image_to_rgba()

    mod = utils.merge_irmodules(
        scale,
        pre_pro,
        rrdb,
        post_pro,
        unscale,
        image_to_rgba
    )
    return relax.frontend.detach_params(mod)


def legalize_and_lift_params(
    mod: tvm.IRModule, model_params: Dict[str, List[tvm.nd.NDArray]], args: Dict
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    model_names = ["rrdb"]
    entry_funcs = ["scale_image", "preprocess", "rrdb", "postprocess", "unscale_image", "image_to_rgba"]

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)
    mod = relax.transform.LiftTransformParams()(mod)

    mod_transform, mod_deploy = utils.split_transform_deploy_mod(
        mod, model_names, entry_funcs
    )

    debug_dump_script(mod_transform, "mod_lift_params.py", args)

    new_params = utils.transform_params(mod_transform, model_params)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy


def build(mod: tvm.IRModule, args: Dict) -> None:
    from tvm import meta_schedule as ms

    db = ms.database.create(work_dir=args.db_path)
    with args.target, db, tvm.transform.PassContext(opt_level=3):
        mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod)
        mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)

    debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target)

    target_kind = args.target.kind.default_keys[0]

    if target_kind == "webgpu":
        output_filename = f"real_esrgan_{target_kind}.wasm"
    else:
        output_filename = f"real_esrgan_{target_kind}.so"

    debug_dump_shader(ex, f"real_esrgan_{target_kind}", args)
    ex.export_library(os.path.join(args.artifact_path, output_filename))


if __name__ == "__main__":
    ARGS = _parse_args()
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    torch_dev_key = utils.detect_available_torch_device()
    cache_path = os.path.join(ARGS.artifact_path, "mod_cache_before_build.pkl")
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    if not use_cache:
        mod, params = trace_models(ARGS)
        mod = legalize_and_lift_params(mod, params, ARGS)
        with open(cache_path, "wb") as outfile:
            pickle.dump(mod, outfile)
        print(f"Save a cached module to {cache_path}.")
    else:
        print(
            f"Load cached module from {cache_path} and skip tracing. "
            "You can use --use-cache=0 to retrace"
        )
        mod = pickle.load(open(cache_path, "rb"))
    build(mod, ARGS)
