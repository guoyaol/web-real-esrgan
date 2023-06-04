#!/bin/bash
set -euxo pipefail

scripts/build_esr_site.sh web/real-esrgan-config.json

echo "symlink parameter location to site.."

ln -s `pwd`/dist/params site/_site/web-sd-shards-v1-5
cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-stable-diffusion --port 8888
