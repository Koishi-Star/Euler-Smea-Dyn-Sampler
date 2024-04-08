# Euler Smea Dyn Sampler
A sampler base on Euler, aim at generating better picture

The Smea sampler can largely avoid structural and limb collapse when generating large images (currently only compatible with square images).

一种基于Euler的采样方法，旨在生成更好的图片

smea采样器可以很大程度上避免出大图时的结构、肢体崩坏,能很大程度得到更优秀的手部（不完美但比已有采样方法更好）

适配绝大多数图片尺寸，在大图的效果尤其优秀，支持完全没训练过的异种尺寸

计算资源消耗：Euler dy将约等于euler a, 而euler smea dy将消耗更多计算资源（约1.25倍）

**how to use**




