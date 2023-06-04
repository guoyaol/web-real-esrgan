let imageProcessed = false; // flag to check if image has been processed

document.getElementById('imgInput').addEventListener('change', function(event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            const imgDisplay = document.getElementById('imgDisplay');
            imgDisplay.style.display = 'block';
            imgDisplay.src = e.target.result;

            imgDisplay.onload = function() {
                if (imageProcessed) return; // skip if image is already processed

                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');

                canvas.width = imgDisplay.width;
                canvas.height = imgDisplay.height;

                context.drawImage(imgDisplay, 0, 0, imgDisplay.width, imgDisplay.height);
                const imageData = context.getImageData(0, 0, imgDisplay.width, imgDisplay.height);

                for(let i = 0; i < imageData.data.length; i += 4) {
                    const r = imageData.data[i];
                    const b = imageData.data[i+2];

                    // swap red and blue channel
                    imageData.data[i] = b;
                    imageData.data[i+2] = r;
                }

                context.putImageData(imageData, 0, 0);
                imgDisplay.src = canvas.toDataURL();

                imageProcessed = true; // set flag to true after processing image
            }
        }

        reader.readAsDataURL(file);
    }
});