from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm

@I.ir_module
class Module:
    @T.prim_func
    def fused_conv2d8_add4_leaky_relu2(lv1783: T.Buffer((T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)), "float32"), self_rrdb_conv_up2_weight: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), lv1785: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(2562), T.int64(1794)))
        var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(2562), T.int64(1794)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv1783[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(2561) and T.int64(1) <= v_i3 and v_i3 < T.int64(1793), lv1783[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(64), T.int64(2560), T.int64(1792), T.int64(64), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], self_rrdb_conv_up2_weight[v_ff, v_rc, v_ry, v_rx])
                T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * self_rrdb_conv_up2_weight[v_ff, v_rc, v_ry, v_rx]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1785[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv1785[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Select(T.float32(0) < var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3] * T.float32(0.20000000000000001))

    @T.prim_func
    def fused_conv2d9_add5(lv1791: T.Buffer((T.int64(1), T.int64(64), T.int64(2560), T.int64(1792)), "float32"), self_rrdb_conv_last_weight: T.Buffer((T.int64(3), T.int64(64), T.int64(3), T.int64(3)), "float32"), lv1793: T.Buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(3), T.int64(2560), T.int64(1792)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(2562), T.int64(1794)))
        var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(2560), T.int64(1792)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(2562), T.int64(1794)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv1791[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(2561) and T.int64(1) <= v_i3 and v_i3 < T.int64(1793), lv1791[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(3), T.int64(2560), T.int64(1792), T.int64(64), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], self_rrdb_conv_last_weight[v_ff, v_rc, v_ry, v_rx])
                T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * self_rrdb_conv_last_weight[v_ff, v_rc, v_ry, v_rx]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(2560), T.int64(1792)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1793[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv1793[v_ax0, v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def fused_conv2d4_add1_leaky_relu(lv17: T.Buffer((T.int64(1), T.int64(160), T.int64(640), T.int64(448)), "float32"), self_rrdb_body_0_rdb1_conv4_weight: T.Buffer((T.int64(32), T.int64(160), T.int64(3), T.int64(3)), "float32"), lv19: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(1)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(640), T.int64(448)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(160), T.int64(642), T.int64(450)))
        var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(640), T.int64(448)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(640), T.int64(448)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(160), T.int64(642), T.int64(450)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv17[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(641) and T.int64(1) <= v_i3 and v_i3 < T.int64(449), lv17[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(32), T.int64(640), T.int64(448), T.int64(160), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], self_rrdb_body_0_rdb1_conv4_weight[v_ff, v_rc, v_ry, v_rx])
                T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * self_rrdb_body_0_rdb1_conv4_weight[v_ff, v_rc, v_ry, v_rx]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(640), T.int64(448)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv19[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv19[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(640), T.int64(448)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Select(T.float32(0) < var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3], var_T_add_intermediate[v_i0, v_i1, v_i2, v_i3] * T.float32(0.20000000000000001))


    @R.function
    def main():
        cls = Module
        lv1 = R.call_tir(cls.fused_conv2d8_add4_leaky_relu2, [], out_sinfo=R.Tensor((), "float32"))
        lv2 = R.call_tir(cls.fused_conv2d9_add5, [], out_sinfo=R.Tensor((), "float32"))
        lv3 = R.call_tir(cls.fused_conv2d4_add1_leaky_relu, [], out_sinfo=R.Tensor((), "float32"))
        return lv1

mod = Module

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
        max_trials_global=15,
        max_trials_per_task=5,
        strategy=ms.search_strategy.EvolutionarySearch(init_min_unmeasured=10, max_fail_count=20),
    )

tune(mod)