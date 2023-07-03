var loadedImageData;
let localRealESRGANInst = new RealESRGANInstance();

document.getElementById('inputImage').addEventListener('change', function (event) {
    loadImageData(event.target.files[0]).then(imageData => {
        loadedImageData = imageData;

        useLoadedImageData();
    });
});

function loadImageData(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, img.width, img.height);
            const imageData = ctx.getImageData(0, 0, img.width, img.height);
            resolve(imageData);
        };
        img.onerror = function() {
            reject(new Error("Failed to load image"));
        };
        img.src = URL.createObjectURL(file);
    });
}

function useLoadedImageData() {
    console.log(loadedImageData);
    localRealESRGANInst.loadImage(loadedImageData); 
}


tvmjsGlobalEnv.asyncOnGenerate = async function () {
await localRealESRGANInst.generate();
};

tvmjsGlobalEnv.asyncOnRPCServerLoad = async function (tvm) {
const inst = new RealESRGANInstance();
await inst.asyncInitOnRPCServerLoad(tvm);
};