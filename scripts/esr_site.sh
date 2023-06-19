#!/bin/bash
set -euxo pipefail

scripts/build_esr_site.sh web/real-esrgan-config.json

echo "symlink parameter location to site.."

ln -s `pwd`/dist/params site/_site/web-eargan-shards
cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-real-esrgan --port 8888
