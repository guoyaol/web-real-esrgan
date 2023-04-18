from network import RRDBNet
import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
import torch
from torch import fx
from tvm import te
from tvm.relax.frontend.torch import from_fx

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

loadnet = torch.load("/home/guoyaol/Real-ESRGAN/weights/RealESRGAN_x4plus.pth", map_location=torch.device('cpu'))


# prefer to use params_ema
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
model.load_state_dict(loadnet[keyname], strict=True)

# fx.symbolic_trace(model).graph.print_tabular()

def rrdb_net(model) -> tvm.IRModule:

    class RRDBNetWrapper(torch.nn.Module):
        def __init__(self, rrdb):
            super().__init__()
            self.rrdb = rrdb

        def forward(self, input):
            output = self.rrdb(input)
            return output

    rrdb = RRDBNetWrapper(model)


    graph = fx.symbolic_trace(rrdb)

    mod = from_fx(
        graph,
        [((1, 3, 128, 128), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})


mod = rrdb_net(model)
