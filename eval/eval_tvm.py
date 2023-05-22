import tvm
from tvm import relax
import cv2
import numpy as np
import os
from typing import Dict, List
import time

target = tvm.target.Target("apple/m2-gpu")
device = tvm.metal()

def load_params(artifact_path: str, device) -> Dict[str, List[tvm.nd.NDArray]]:
    from tvm.contrib import tvmjs

    pdict = {}
    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    for model in ["rrdb"]:
        plist = []
        size = meta[f"{model}ParamSize"]
        for i in range(size):
            plist.append(params[f"{model}_{i}"])
        pdict[model] = plist
    return pdict

const_params_dict = load_params(artifact_path="../dist", device=device)
ex = tvm.runtime.load_module("../dist/real_esrgan.so")

vm = relax.VirtualMachine(rt_mod=ex, device=device)

class TVMESRPipeline:
    def __init__(
        self,
        vm: relax.VirtualMachine,
        tvm_device,
        param_dict,
    ):
        def wrapper(f, params):
            def wrapped_f(*args):
                return f(*args, params)

            return wrapped_f

        self.vm = vm
        
        self.rrdb = wrapper(vm["rrdb"], param_dict["rrdb"])
        
        self.scale_image = vm["scale_image"]
        self.preprocess = vm["preprocess"]
        self.unscale_image = vm["unscale_image"]
        self.postprocess = vm["postprocess"]

        self.tvm_device = tvm_device
        self.param_dict = param_dict

    def __call__(self, input_image: np.array):
        image = tvm.nd.array(input_image, device=self.tvm_device)
        image = self.scale_image(image)
        image = self.preprocess(image)
        image = self.rrdb(image)
        image = self.postprocess(image)
        image = self.unscale_image(image)

        return image
    

pipe = TVMESRPipeline(vm, device, const_params_dict)

input_path = "../input/OST_009.png"
output_path = "./output"

imgname, extension = os.path.splitext(os.path.basename(input_path))
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

start_time = time.time()

for i in range(10):
    img = img.astype(np.float32)

    output = pipe(img)

end_time = time.time()

execution_time = end_time - start_time  # subtract start_time from end_time
print(f"Executed the code in: {execution_time} seconds")  # print the execution time

final_result = output.numpy().astype(np.uint8)
cv2.imwrite("./output/TVM_output.png", final_result)