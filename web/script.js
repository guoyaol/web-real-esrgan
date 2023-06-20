class ImageConverter {
    constructor() {
        this.imageUpload = document.getElementById('imageUpload');
        this.convertButton = document.getElementById('convertButton');
        this.canvas = document.getElementById('canvas');
        this.context = this.canvas.getContext('2d');
        this.img = null;
    }

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

    convertImage() {
        if (!this.img) return;
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.context.drawImage(this.img, 0, 0, this.img.width, this.img.height);
        let imageData = this.context.getImageData(0, 0, this.img.width, this.img.height);
        for (let i = 0; i < imageData.data.length; i += 4) {
            let red = imageData.data[i];
            let blue = imageData.data[i + 2];
            // swap red and blue
            imageData.data[i] = blue;
            imageData.data[i + 2] = red;
        }
        this.context.putImageData(imageData, 0, 0);
    }
}

window.onload = function() {
    window.imageConverter = new ImageConverter();
};
