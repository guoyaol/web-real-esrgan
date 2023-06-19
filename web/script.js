window.onload = function() {
    const imageUpload = document.getElementById('imageUpload');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    imageUpload.addEventListener('change', function() {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                context.drawImage(img, 0, 0, img.width, img.height);
                let imageData = context.getImageData(0, 0, img.width, img.height);
                console.log(imageData);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    let red = imageData.data[i];
                    let blue = imageData.data[i + 2];
                    // swap red and blue
                    imageData.data[i] = blue;
                    imageData.data[i + 2] = red;
                }
                context.putImageData(imageData, 0, 0);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(this.files[0]);
    }, false);
};
