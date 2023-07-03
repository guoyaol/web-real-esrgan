//TODO: how to imoport from src/RealEsrgan.js?

var loadedImageData;
let localRealESRGANInst = new RealESRGANInstance();

document.getElementById('inputImage').addEventListener('change', function (event) {
    loadImageData(event.target.files[0]).then(imageData => {
        loadedImageData = imageData;

        // Now that we've loaded the image data, we can use the loadedImageData variable.
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

// This is a function in the global scope that uses loadedImageData.
function useLoadedImageData() {
    console.log(loadedImageData);
    localRealESRGANInst.loadImage(loadedImageData); 
    // Here, you can add any other code that uses loadedImageData.
    // Because this function is only called after an image has been loaded,
    // you can be sure that loadedImageData will not be undefined.
}

  

// tvmjsGlobalEnv.loadImage = function (event) {
// localRealESRGANInst.loadImage(event);
// };

tvmjsGlobalEnv.asyncOnGenerate = async function () {
await localRealESRGANInst.generate();
};

tvmjsGlobalEnv.asyncOnRPCServerLoad = async function (tvm) {
const inst = new RealESRGANInstance();
await inst.asyncInitOnRPCServerLoad(tvm);
};