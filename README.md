# Overall_sampling
A sampler base on Euler, aim at generating better picture

The Smea sampler can largely avoid structural and limb collapse when generating large images (currently only compatible with square images).

一种基于Euler的采样方法，旨在生成更好的图片

smea采样器可以很大程度上避免出大图时的结构、肢体崩坏（目前只适配方图）

**how to use**

download sampling.py, put it in sd-webui-aki-v4.5\repositories\k-diffusion\k_diffusion and replace, don't foget to backup.

download sd_samplers_kdiffusion.py, put it in d-webui-aki-v4.5\modules and replace, don't foget to backup.

**2024.04.02**  
灵感源自于nai3的dyn和smea采样器。

目前完成：smea:alpha 版本，dyn:尚未完成。

消耗的计算资源：约等于euler a。

Inspired by the Nai3's dyn and smea samplers.

Currently completed: smea:alpha version, dyn: not yet completed. 

Computational resources consumed: approximately equal to euler a. 

有关smea alpha的一个简单对比：

A simple comparison regarding smea alpha:

Euler Smea： ![00021](https://github.com/Koishi-Star/Overall_sampling/assets/66173435/c4591528-8937-440c-8fd9-e0cf4eb955b7)

Euler a： ![00020](https://github.com/Koishi-Star/Overall_sampling/assets/66173435/576b4173-cbdb-4935-bfd7-38afbc990116)

Metadata：1girl,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue eyes,grey hair,short hair,bangs,hair between eyes,eyebrows visible through hair,blush,closed mouth,smile,neck ribbon,white sleeveless dress,crease,frilled_collar,detached_sleeves,flat chest,outdoors,river,
Negative prompt: EasyNegative,
Steps: 20, Sampler: Euler Smea, CFG scale: 7, Seed: 4265333208, Size: 1024x1024, Model hash: 54ef3e3610, Model: meinamix_meinaV11, Clip skip: 2, TI hashes: "EasyNegative: c74b4e810b03", Version: v1.8.0


Euler Smea： ![00019](https://github.com/Koishi-Star/Overall_sampling/assets/66173435/8b3f3ea5-6b79-402b-8524-f86bc424d9a0)

Euler a: ![00018](https://github.com/Koishi-Star/Overall_sampling/assets/66173435/d0e3a96f-175f-49ba-8ec7-8a8566010e09)

MetaData：1girl,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue eyes,grey hair,short hair,bangs,hair between eyes,eyebrows visible through hair,blush,closed mouth,smile,neck ribbon,white sleeveless dress,crease,frilled_collar,detached_sleeves,flat chest,outdoors,river,
Negative prompt: EasyNegative,
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 4265333208, Size: 1024x1024, Model hash: 39d6af08b2, Model: qteamixQ_omegaFp16, Clip skip: 2, ENSD: 1337, Eta: 0.35, Version: v1.8.0


