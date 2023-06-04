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
   *
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
    const latentShape = [1, 4, 64, 64];

    var unetNumSteps;
    if (schedulerId == 0) {
      scheduler = new TVMDPMSolverMultistepScheduler(
        this.schedulerConsts[0], latentShape, this.tvm, this.device, this.vm);
      unetNumSteps = this.schedulerConsts[0]["num_steps"];
    } else {
      scheduler = new TVMPNDMScheduler(
        this.schedulerConsts[1], latentShape, this.tvm, this.device, this.vm);
      unetNumSteps = this.schedulerConsts[1]["num_steps"];
    }
    const totalNumSteps = unetNumSteps + 2;

    if (progressCallback !== undefined) {
      progressCallback("clip", 0, 1, totalNumSteps);
    }

    const embeddings = this.tvm.withNewScope(() => {
      let posInputIDs = this.tokenize(prompt);
      let negInputIDs = this.tokenize(negPrompt);
      const posEmbeddings = this.clipToTextEmbeddings(
        posInputIDs, this.clipParams);
      const negEmbeddings = this.clipToTextEmbeddings(
        negInputIDs, this.clipParams);
      // maintain new latents
      return this.tvm.detachFromCurrentScope(
        this.concatEmbeddings(negEmbeddings, posEmbeddings)
      );
    });
    // use uniform distribution with same variance as normal(0, 1)
    const scale = Math.sqrt(12) / 2;
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );
    this.tvm.endScope();
    //---------------------------
    // Stage 1: UNet + Scheduler
    //---------------------------
    if (vaeCycle != -1) {
      // show first frame
      this.tvm.withNewScope(() => {
        const image = this.vaeToImage(latents, this.vaeParams);
        this.tvm.showImage(this.imageToRGBA(image));
      });
      await this.device.sync();
    }
    vaeCycle = vaeCycle == -1 ? unetNumSteps : vaeCycle;
    let lastSync = undefined;

    for (let counter = 0; counter < unetNumSteps; ++counter) {
      if (progressCallback !== undefined) {
        progressCallback("unet", counter, unetNumSteps, totalNumSteps);
      }
      const timestep = scheduler.timestep[counter];
      // recycle noisePred, track latents manually
      const newLatents = this.tvm.withNewScope(() => {
        this.tvm.attachToCurrentScope(latents);
        const noisePred = this.unetLatentsToNoisePred(
          latents, timestep, embeddings, this.unetParams);
        // maintain new latents
        return this.tvm.detachFromCurrentScope(
          scheduler.step(noisePred, latents, counter)
        );
      });
      latents = newLatents;
      // use skip one sync, although likely not as useful.
      if (lastSync !== undefined) {
        await lastSync;
      }
      // async event checker
      lastSync = this.device.sync();

      // Optionally, we can draw intermediate result of VAE.
      if ((counter + 1) % vaeCycle == 0 &&
        (counter + 1) != unetNumSteps &&
        counter >= beginRenderVae) {
        this.tvm.withNewScope(() => {
          const image = this.vaeToImage(latents, this.vaeParams);
          this.tvm.showImage(this.imageToRGBA(image));
        });
        await this.device.sync();
      }
    }
    scheduler.dispose();
    embeddings.dispose();
    //-----------------------------
    // Stage 2: VAE and draw image
    //-----------------------------
    if (progressCallback !== undefined) {
      progressCallback("vae", 0, 1, totalNumSteps);
    }
    this.tvm.withNewScope(() => {
      const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(image));
    });
    latents.dispose();
    await this.device.sync();
    if (progressCallback !== undefined) {
      progressCallback("vae", 1, 1, totalNumSteps);
    }
  }

  clearCanvas() {
    this.tvm.clearCanvas();
  }
};
