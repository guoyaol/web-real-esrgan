class RealESRGANPipeline {
  constructor(tvm, cacheMetadata) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
    this.tvm = tvm;

    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("canvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.rrdbResNet = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("rrdb")
    );
    this.rrdbParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("rrdb", cacheMetadata.clipParamSize)
    );

    this.scale = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("scale")
    );

    this.unscale = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unscale")
    );

    this.preprocess = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("preprocess")
    );

    this.postprocess = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("postprocess")
    );
  }

  dispose() {
    // note: tvm instance is not owned by this class
    this.rrdbParams.dispose();
    this.rrdbResNet.dispose();
    this.scale.dispose();
    this.unscale.dispose();
    this.preprocess.dispose();
    this.postprocess.dispose();
  }

  /**
   * async preload webgpu pipelines when possible.
   */
  async asyncLoadWebGPUPiplines() {
    await this.tvm.asyncLoadWebGPUPiplines(this.vm.getInternalModule());
  }

  //TODO: add web ESRGAN generation pipeline
  /**
   * Run generation pipeline.
   * @param inputImage Input image.
   * @param prompt Input prompt.
   * @param negPrompt Input negative prompt.
   * @param progressCallback Callback to check progress.
   * @param schedulerId The integer ID of the scheduler to use.
   * - 0 for multi-step DPM solver,
   * - 1 for PNDM solver.
   * @param vaeCycle optionally draw VAE result every cycle iterations.
   * @param beginRenderVae Begin rendering VAE after skipping these warmup runs.
   */
  async generate(
    inputImage,
    prompt,
    negPrompt = "",
    progressCallback = undefined,
    schedulerId = 0,
    vaeCycle = -1,
    beginRenderVae = 10
  ) {
    // Principle: beginScope/endScope in synchronized blocks,
    // this helps to recycle intermediate memories
    // detach states that needs to go across async boundaries.
    //--------------------------
    // Stage 0: CLIP
    //--------------------------
    this.tvm.beginScope();
    // get latents
    const latentShape = [1, 3, 640, 448];
    // use uniform distribution with same variance as normal(0, 1)
    const scale = Math.sqrt(12) / 2;
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );
    this.tvm.endScope();

    this.tvm.withNewScope(() => {
      const scaledImage = this.scale(latents);
      const preImage = this.preprocess(scaledImage);
      const rrdbImage = this.rrdbResNet(preImage, this.rrdbParams);
      const postImage = this.postprocess(rrdbImage);
      const outImage = this.unscale(postImage);

      // const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(outImage));
    });
    // latents.dispose();
    await this.device.sync();
    // if (progressCallback !== undefined) {
    //   progressCallback("vae", 1, 1, totalNumSteps);
    // }
  }

  clearCanvas() {
    this.tvm.clearCanvas();
  }
};
