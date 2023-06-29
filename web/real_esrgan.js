class RealESRGANPipeline {
  constructor(tvm, cacheMetadata) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
    this.tvm = tvm;

    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("tcanvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.rrdbResNet = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("rrdb")
    );
    this.rrdbParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("rrdb", cacheMetadata.rrdbParamSize)
    );

    this.scale = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("scale_image")
    );

    this.unscale = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unscale_image")
    );

    this.preprocess = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("preprocess")
    );

    this.postprocess = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("postprocess")
    );

    this.imageToRGBA = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("image_to_rgba")
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
    this.imageToRGBA.dispose();
  }

  /**
   * async preload webgpu pipelines when possible.
   */
  async asyncLoadWebGPUPiplines() {
    await this.tvm.asyncLoadWebGPUPiplines(this.vm.getInternalModule());
  }

  //TODO: add web ESRGAN generation pipeline
  /**
   * @param lowImage Begin rendering VAE after skipping these warmup runs.
   */
  async generate(lowImage) {
    // Principle: beginScope/endScope in synchronized blocks,
    // this helps to recycle intermediate memories
    // detach states that needs to go across async boundaries.
    //--------------------------
    // Stage 0: CLIP
    //--------------------------
    this.tvm.beginScope();
    // get latents
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.empty([160, 160, 3], "float32", this.tvm.webgpu()).copyFrom(lowImage)
      );
    this.tvm.endScope();

    console.log(latents)

    this.tvm.withNewScope(() => {
      const scaledImage = this.scale(latents);
      const preImage = this.preprocess(scaledImage);
      // console.log(preImage)
      const rrdbImage = this.rrdbResNet(preImage, this.rrdbParams);
      // console.log(rrdbImage)
      // const outShape = [1, 3, 716, 716];
      // const rrdbImage  = this.tvm.uniform(outShape, 0.1, 0.5, this.tvm.webgpu());
      const postImage = this.postprocess(rrdbImage);
      const outImage = this.unscale(postImage);
      // console.log(postImage);
      const showImage = this.imageToRGBA(outImage);
      // console.log(showImage);

      // const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(showImage);
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

class RealESRGANInstance {
  constructor() {
    this.tvm = undefined;
    this.pipeline = undefined;
    this.config = undefined;
    this.generateInProgress = false;
    this.logger = console.log;
    this.imageUpload = document.getElementById('imageUpload');
    this.convertButton = document.getElementById('convertButton');
    this.canvas = document.getElementById('canvas');
    this.context = this.canvas.getContext('2d');
    this.img = null;
  }

  //TODO: remove from esrgan
  loadImage(event) {
    const reader = new FileReader();
    reader.onload = (event) => {
        this.img = new Image();
        this.img.onload = () => {
          this.canvas.width = this.img.width;
          this.canvas.height = this.img.height;
          this.convertButton.disabled = false;
        };
        this.img.src = event.target.result;
    };
    reader.readAsDataURL(event.target.files[0]);
  }

  /**
   * Initialize TVM
   * @param wasmUrl URL to wasm source.
   * @param cacheUrl URL to NDArray cache.
   * @param logger Custom logger.
   */
  async #asyncInitTVM(wasmUrl, cacheUrl) {
    if (this.tvm !== undefined) {
      return;
    }

    //TODO: not important
    if (document.getElementById("log") !== undefined) {
      this.logger = function (message) {
        console.log(message);
        const d = document.createElement("div");
        d.innerHTML = message;
        document.getElementById("log").appendChild(d);
      };
    }

    const wasmSource = await (
      await fetch(wasmUrl)
    ).arrayBuffer();
    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      new EmccWASI(),
      this.logger
    );
    // initialize WebGPU
    try {
      const output = await tvmjs.detectGPUDevice();
      if (output !== undefined) {
        var label = "WebGPU";
        if (output.adapterInfo.description.length != 0) {
          label += " - " + output.adapterInfo.description;
        } else {
          label += " - " + output.adapterInfo.vendor;
        }
        document.getElementById(
          "gpu-tracker-label").innerHTML = ("Initialize GPU device: " + label);
        tvm.initWebGPU(output.device);
      } else {
        document.getElementById(
          "gpu-tracker-label").innerHTML = "This browser env do not support WebGPU";
        this.reset();
        throw Error("This browser env do not support WebGPU");
      }
    } catch (err) {
      document.getElementById("gpu-tracker-label").innerHTML = (
        "Find an error initializing the WebGPU device " + err.toString()
      );
      console.log(err.stack);
      this.reset();
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }

    this.tvm = tvm;
    function initProgressCallback(report) {
      document.getElementById("progress-tracker-label").innerHTML = report.text;
      document.getElementById("progress-tracker-progress").value = report.progress * 100;
    }
    tvm.registerInitProgressCallback(initProgressCallback);
    if (!cacheUrl.startsWith("http")) {
      cacheUrl = new URL(cacheUrl, document.URL).href;
    }
    await tvm.fetchNDArrayCache(cacheUrl, tvm.webgpu());
  }

  /**
   * Initialize the pipeline
   *
   */
  async #asyncInitPipeline() {
    if (this.tvm == undefined) {
      throw Error("asyncInitTVM is not called");
    }
    if (this.pipeline !== undefined) return;
    //TODO: delete tokenizer
    // const tokenizer = await tvmjsGlobalEnv.getTokenizer(tokenizerName);
    this.pipeline = this.tvm.withNewScope(() => {
      return new RealESRGANPipeline(this.tvm, this.tvm.cacheMetadata);
    });
    await this.pipeline.asyncLoadWebGPUPiplines();
  }

  /**
   * Async initialize config
   */
  async #asyncInitConfig() {
    if (this.config !== undefined) return;
    this.config = await (await fetch("real-esrgan-config.json")).json();
  }

  /**
   * Function to create progress callback tracker.
   * @returns A progress callback tracker.
   */
  #getProgressCallback() {
    const tstart = performance.now();
    function progressCallback(stage, counter, numSteps, totalNumSteps) {
      const timeElapsed = (performance.now() - tstart) / 1000;
      let text = "Generating ... at stage " + stage;
      if (stage == "unet") {
        counter += 1;
        text += " step [" + counter + "/" + numSteps + "]"
      }
      if (stage == "vae") {
        counter = totalNumSteps;
      }
      text += ", " + Math.ceil(timeElapsed) + " secs elapsed.";
      document.getElementById("progress-tracker-label").innerHTML = text;
      document.getElementById("progress-tracker-progress").value = (counter / totalNumSteps) * 100;
    }
    return progressCallback;
  }

  /**
   * Async initialize instance.
   */
  async asyncInit() {
    if (this.pipeline !== undefined) return;
    await this.#asyncInitConfig();
    await this.#asyncInitTVM(this.config.wasmUrl, this.config.cacheUrl);
    await this.#asyncInitPipeline();
  }

  /**
   * Async initialize
   *
   * @param tvm The tvm instance.
   */
  async asyncInitOnRPCServerLoad(tvmInstance) {
    if (this.tvm !== undefined) {
      throw Error("Cannot reuse a loaded instance for rpc");
    }
    this.tvm = tvmInstance;

    this.tvm.beginScope();
    this.tvm.registerAsyncServerFunc("generate", async (lowImage) => {
      // this.lowImage = lowImage;
      await this.pipeline.generate(lowImage);
    });
    this.tvm.registerAsyncServerFunc("clearCanvas", async () => {
      this.tvm.clearCanvas();
    });
    this.tvm.registerAsyncServerFunc("showImage", async (data) => {
      this.tvm.showImage(data);
    });
    this.tvm.endScope();
  }

  /**
   * Run generate
   */
  async generate() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.drawImage(this.img, 0, 0, this.img.width, this.img.height);
    let imageData = this.context.getImageData(0, 0, this.img.width, this.img.height);
    console.log(imageData);
    const unit8Array = imageData.data;

    const rgbArray = [];

    for (let i = 0; i < unit8Array.length; i += 4) {
        rgbArray.push(unit8Array[i]);     // R value
        rgbArray.push(unit8Array[i + 1]); // G value
        rgbArray.push(unit8Array[i + 2]); // B value
        // skipping rgbaArray[i + 3] because that's the A value
    }

    const float32Array = Float32Array.from(rgbArray);

    if (this.requestInProgress) {
      this.logger("Request in progress, generate request ignored");
      return;
    }
    this.requestInProgress = true;
    try {
      await this.asyncInit();
      await this.pipeline.generate(float32Array);
    } catch (err) {
      this.logger("Generate error, " + err.toString());
      console.log(err.stack);
      this.reset();
    }
    this.requestInProgress = false;
  }

  /**
   * Reset the instance;
   */
  reset() {
    this.tvm = undefined;
    if (this.pipeline !== undefined) {
      this.pipeline.dispose();
    }
    this.pipeline = undefined;
  }
}


localRealESRGANInst = new RealESRGANInstance();

tvmjsGlobalEnv.loadImage = function (event) {
  localRealESRGANInst.loadImage(event);
};

tvmjsGlobalEnv.asyncOnGenerate = async function () {
  await localRealESRGANInst.generate();
};

tvmjsGlobalEnv.asyncOnRPCServerLoad = async function (tvm) {
  const inst = new RealESRGANInstance();
  await inst.asyncInitOnRPCServerLoad(tvm);
};