//TODO: how to imoport from src/RealEsrgan.js?
  
let localRealESRGANInst = new RealESRGANInstance();

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