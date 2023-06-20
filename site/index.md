---
layout: default
title: Home
notitle: true
---



### Demo

{% include real_esrgan.html %}

<!-- ### Script

{% include ref.html %} -->


### Notes

* WebGPU spec does comes with FP16 support already, but the implementation does not yet support this feature at this moment. As a result, the memory consumption of running the demo is about 7GB. For Apple silicon Mac with only 8GB of unified memory, it may take longer (a few minutes) to generate an image. This demo may also work for Mac with AMD GPU.
* Please check out our [GitHub repo](https://github.com/mlc-ai/web-stable-diffusion) for running the same shader flow locally on your GPU device through the native driver. Right now, there are still gaps (e.g., without launching Chrome from command line, Chrome’s WebGPU implementation inserts bound clips for all array index access, such that `a[i]` becomes `a[min(i, a.size)]`, which are not optimized out by the downstream shader compilers), but we believe it is feasible to close such gaps as WebGPU dispatches to these native drivers.

## Disclaimer

This demo site is for research purposes only. Please conform to the [uses of stable diffusion models](https://huggingface.co/runwayml/stable-diffusion-v1-5#uses).
