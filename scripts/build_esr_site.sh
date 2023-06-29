#!/bin/bash
set -euxo pipefail

if [[ ! -f $1 ]]; then
    echo "cannot find config" $1
fi

rm -rf site/dist
mkdir -p site/dist site/_inlcudes

echo "Copy local configurations.."
cp $1 site/real-esrgan-config.json
echo "Copy files..."
cp web/real_esrgan.html site/_includes
cp web/real_esrgan.js site/dist
cp web/script.js    site/dist

cp dist/real_esrgan_webgpu.wasm site/dist

cp dist/tvmjs_runtime.wasi.js site/dist
cp dist/tvmjs.bundle.js site/dist

cd site && jekyll b && cd ..