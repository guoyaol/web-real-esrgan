# fmt: off

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ir as I, relax as R, tir as T
import numpy as np


@I.ir_module
class fused_matmul1_add1:
    @T.prim_func
    def main(lv1778: T.Buffer((T.int64(1), T.int64(64), T.int64(1280), T.int64(896)), "float32"), self_rrdb_conv_up1_weight: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), lv1780: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(64), T.int64(1280), T.int64(896)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1282), T.int64(898)))
        var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1280), T.int64(896)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1280), T.int64(896)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(1125)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) // T.int64(1151236))
                        v_i2 = T.axis.spatial(T.int64(1282), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) % T.int64(1151236) // T.int64(898))
                        v_i3 = T.axis.spatial(T.int64(898), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) % T.int64(898))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(256) + i0_i1_i2_i3_fused_2 < T.int64(73679104))
                        T.reads(lv1778[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1281) and T.int64(1) <= v_i3 and v_i3 < T.int64(897), lv1778[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(1120), T.int64(64), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(64), (nn_ff_yy_xx_fused_0 * T.int64(65536) + nn_ff_yy_xx_fused_1 * T.int64(256) + nn_ff_yy_xx_fused_2) // T.int64(1146880))
                        v_yy = T.axis.spatial(T.int64(1280), (nn_ff_yy_xx_fused_0 * T.int64(65536) + nn_ff_yy_xx_fused_1 * T.int64(256) + nn_ff_yy_xx_fused_2) % T.int64(1146880) // T.int64(896))
                        v_xx = T.axis.spatial(T.int64(896), (nn_ff_yy_xx_fused_0 * T.int64(65536) + nn_ff_yy_xx_fused_1 * T.int64(256) + nn_ff_yy_xx_fused_2) % T.int64(896))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], self_rrdb_conv_up1_weight[v_ff, v_rc, v_ry, v_rx])
                        T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                        var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * self_rrdb_conv_up1_weight[v_ff, v_rc, v_ry, v_rx]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1120)):
                    with T.block("T_add"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(65536) + ax0_ax1_ax2_ax3_fused_1 * T.int64(256) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1146880))
                        v_ax2 = T.axis.spatial(T.int64(1280), (ax0_ax1_ax2_ax3_fused_0 * T.int64(65536) + ax0_ax1_ax2_ax3_fused_1 * T.int64(256) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1146880) // T.int64(896))
                        v_ax3 = T.axis.spatial(T.int64(896), (ax0_ax1_ax2_ax3_fused_0 * T.int64(65536) + ax0_ax1_ax2_ax3_fused_1 * T.int64(256) + ax0_ax1_ax2_ax3_fused_2) % T.int64(896))
                        T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1780[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                        var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv1780[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(1120)):
                    with T.block("compute"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) // T.int64(1146880))
                        v_i2 = T.axis.spatial(T.int64(1280), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) % T.int64(1146880) // T.int64(896))
                        v_i3 = T.axis.spatial(T.int64(896), (i0_i1_i2_i3_fused_0 * T.int64(65536) + i0_i1_i2_i3_fused_1 * T.int64(256) + i0_i1_i2_i3_fused_2) % T.int64(896))
                        T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                        T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                        var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Select(T.float32(0) < var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3] * T.float32(0.20000000000000001))

mod = fused_matmul1_add1
# print(tvm.lower(mod["main"]))


target = tvm.target.Target("apple/m2-gpu")
# target = tvm.target.Target("llvm")
# target = tvm.target.Target("cuda")
f = tvm.build(mod["main"], target=target)

print(f.imported_modules[0].get_source())

np.random.seed(0)
np1 = np.random.rand(1, 64, 1280, 896).astype("float32")
np2 = np.random.rand(64, 64, 3, 3).astype("float32")
np3 = np.random.rand(1, 64, 1, 1).astype("float32")
np4 = np.random.rand(1, 64, 1280, 896).astype("float32")

device = tvm.metal()
# device = tvm.cpu()
# device = tvm.cuda()
x1 = tvm.nd.array(np1, device=device)
x2 = tvm.nd.array(np2, device=device)
x3 = tvm.nd.array(np3, device=device)
x4 = tvm.nd.array(np4, device=device)
x = [x1, x2, x3, x4]

f(*x)

print("pass")