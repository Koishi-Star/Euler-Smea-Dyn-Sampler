| Catalog |
| ---- |
| [2024.05.09 Add Euler Negative And Euler dy Negtive 添加两个新采样器](#section1) |
| [2024.04.24](#section2) |
| [2024.04.18 Stage Technical Report 阶段性技术报告](#section3) |
| [2024.04.15 Compatible with Stable Cascade Models 适配Sc模型](#section4) |
| [2024.04.13 Fix Bug In ComfyUI 修复ComfyUI中的Bug](#section5) |
| [2024.04.11 Important! This repo can be use as a extension! 重大！现在此仓库可作为插件使用](#section6) |
| [2024.04.10](#section7) |
| [2024.04.09](#section8) |
| [Euler Smea Dyn Sampler](#section9) |
| [Effect 效果](#section10) |
| [how to use(This has become outdated, but it will be retained) 如何使用（已经过期，依然保留，仅做参考）](#section11) |
| [The technical principles 技术报告](#section12) |
| [Contact the author 联系作者](#section13) |


<a id="section1"></a>
## 2024.05.09 Add Euler Negative And Euler dy Negtive

新增两个采样器，Euler Negative 和 Euler dy Negtive。我不会说它们效果比别的好，因为没有理论依据。不过在实践中我很喜欢它们。

在SDXL表现更好一些，但在SD1.5使用效果也不差

我得去稍微进修一下关于AI的知识，目前这种纯粹依靠灵感和实践的方案过于自由。

以下是关于它们的测试：

Two new samplers have been added, Euler Negative and Euler dy Negative. I won't claim they perform better than others because there's no theoretical basis for it. However, in practice, I quite like them.

They perform slightly better in SDXL, but their performance in SD1.5 is also decent.

I need to brush up on my professional knowledge of AI. Currently, relying solely on intuition and practice feels too unrestricted.

Below are the test results for them:

**768x768， model meinaMixV11**
![xyz_grid-0005-1234-1girl,heart hands,river,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue eyes,grey hair,short hair,bangs,h](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/04131413-0173-4afc-91dc-d5ac42d0560c)

**832x1216，model kohaku-xl-epsilon**
![xyz_grid-0007-4286407380-1girl,_(midori _(blue archive_)_1 1),blue archive,_ciloranko,lobelia _(saclia_),(konya karasue_0 9),wanke,(jiu ye sang_1 1),(rum](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/c237d930-cb00-4142-9b46-faeb59c4faba)
![xyz_grid-0008-3096426698-1girl,_mayano top gun _(umamusume_),umamusume,_shiro9jira,ciloranko,ask _(askzy_),(tianliang duohe fangdongye_0 8),_(solo_1 2),(](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/6d47de71-ceca-4091-8dac-3c59bb1aee1e)
![xyz_grid-0016-789654357-1girl,_gotoh hitori,bocchi the rock!,_ciloranko,maccha _(mochancc_),lobelia _(saclia_),migolu,ask _(askzy_),wanke,(jiu ye sang_1](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/f46be012-2b07-4408-9449-b5a20769ebb4)

**832x1216，model animegineV30**
![xyz_grid-0014-3532334-1girl,_vivlos _(umamusume_),umamusume,_ciloranko,maccha _(mochancc_),lobelia _(saclia_),migolu,ask _(askzy_),wanke,(jiu ye sang_](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/92bd48bf-fe30-45b6-9014-5f98bd3622d7)

**Please Note**

目前插件有一些小bug，会使得人物在画面中的占比变小，就像这样：

The current plugin has a few minor bugs that cause the characters to shrink in the frame, like this:

![xyz_grid-0011-3096426698-1girl,_mayano top gun _(umamusume_),umamusume,_shiro9jira,ciloranko,ask _(askzy_),(tianliang duohe fangdongye_0 8),_(solo_1 2),(](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/41d067ec-7e49-413c-bc4a-9e8e8b173140)
![xyz_grid-0013-3096426698-1girl,_mayano top gun _(umamusume_),umamusume,_shiro9jira,ciloranko,ask _(askzy_),(tianliang duohe fangdongye_0 8),_(solo_1 2),(](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/80c1481a-d5a6-40c1-96a6-665f215c7b39)

所以你可以采用其他方案，例如修改源码的方案去添加这两个采样器，请参考：[How to use](#section11)

So, you can consider alternative solutions, such as modifying the source code to add these two samplers. Please refer to:[How to use](#section11)

<a id="section2"></a>
## 2024.04.24

简单分析Dy Step的原理

最近我尝试在https://civitai.com/models/399873/kohaku-xl-epsilon上测试了Euler Dy，效果不尽人意（但在ang3以及pony系列效果不错）。因此我咨询了作者，得到的回复是“模型没有使用任何低分辨率的图片进行训练”。我想这就是原因所在，Euler Dy将图片放在一个小的尺度上，让去噪工作来到模型的舒适区，并给予一个参考。尤其在SD1.5，Euler Dy确保图像始终处于模型的舒适区域。

而在本次的SDXL模型中，它几乎遗忘了如何在小尺度上生成图片。所以，Dy Step的改进方向已经变得很明显：寻找到SDXL模型的舒适区域，并让采样器在SDXL的舒适区工作。

同时我也写了几个其他的采样器，效果平平，达不到Dy Step的效果。如果有人想试试它们，请在评论区留言。

Recently, I attempted to analyze the principle of Dy Step on https://civitai.com/models/399873/kohaku-xl-epsilon using Euler Dy. The results were unsatisfactory (although they performed well on ang3 and pony series). Therefore, I consulted the author and received the response that "the model did not use any low-resolution images for training." I believe this is the reason why. Euler Dy places images on a small scale, allowing denoising to operate within the model's comfort zone and providing a reference. Especially in SD1.5, Euler Dy ensures that the image always remains within the model's comfort zone.

However, in the current SDXL model, it has almost forgotten how to generate images on a small scale. Therefore, the direction for improving Dy Step has become apparent: to find the comfort zone of the SDXL model and enable the sampler to work within the comfort zone of SDXL.

I've also written a few other samplers, but their performance is mediocre and doesn't match up to Dy Step's effectiveness. If anyone wants to try them out, please leave a comment in the discussions.

<a id="section3"></a>
## 2024.04.18 Stage Technical Report

阶段性技术报告报告。

这些日子里，我尝试了超过二十种策略，但采样器的质量总是优于euler a却差于euler dy，所以暂时还不能发布euler dy a。我必须承认这和nai3的dyn是不同的东西。我依旧会长期维护这个项目，并为了新的采样方法努力，同时尽可能降低ai的算力需求。

Stage Technical Report

In these days, I have attempted over twenty strategies, but the quality of the sampler is always better than Euler A yet worse than Euler DY. Therefore, I cannot release Euler DY A for the time being. I must acknowledge that this is different from NAI3's DYN. I will continue to maintain this project in the long term and work towards developing new sampling methods while trying to minimize the AI's computing power requirements as much as possible.

![xyz_grid-0036-1234-1girl,heart hands,river,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue eyes,grey hair,short hair,bangs,h](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/32618333-0228-472e-bfbe-6c1a2e84ae96)


<a id="section4"></a>
## 2024.04.15 Compatible with Stable Cascade Models

Makes dy_step respect original channel count, making it compatible with Stable Cascade models.

使dy_step遵循原本的通道数，使其与Stable Cascade模型相适应。

<a id="section5"></a>
## 2024.04.13 Fix Bug In ComfyUI

Change code for ComfyUI import. This will fix the overwrite error that occurs in ComfyUI when other extensions use `scripts` as the import folder (I really hope ComfyUI will standardize its interfaces and version dependencies).

P.S.You may find some commits with no means, that because I am not familiar with Github, and try times. So don't care.

更改代码，用于ComfyUI导入。这将修复在ComfyUI中存在其他插件时，若其他插件将`scripts`作为导入文件夹时引起的覆盖错误。（真希望ComfyUI能规范一下它的接口和版本依赖）。

P.S.你可能会发现一些无意义的提交，这是因为我不熟悉Github的使用，并且尝试了几次。别在意。

<a id="section6"></a>
## 2024.04.11 Important! This repo can be use as a extension!

Thanks for @pamparamm, his selfless work has been a great help.

Now this sampler can be use as a extension for **ComfyUI** and **WebUI from Automatic1111**.

The inpainting bug will be fixed.(**At least doesn't throw any exceptions.**)

Thanks again.

Another extension from @licyk , in repo: https://github.com/licyk/advanced_euler_sampler_extension **suitable for 1.8 version**

It's also useful, and thanks hard efforts from licky, too.

In the future, I will work on making dy step compatible with more samplers (such as the DPM series).

感谢 @pamparamm，他的无私工作帮助很大。

现在，这个采样器可以作为 **ComfyUI** 和 **Automatic1111 的 WebUI** 的扩展来使用。

修复了inpainting的bug。（**至少不再抛出异常。**）

再次感谢。

另一个拓展来自@licyk，位于： https://github.com/licyk/advanced_euler_sampler_extension **适用于1.8**

也同样很好用， 同样感谢licyk的辛勤努力。

之后我会想办法让dy step适配更多采样器（例如dpm系列）。

<a id="section7"></a>
## 2024.04.10

![image](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/aa3dd88a-5760-4589-857c-5717a3253ea0)

Find a way to avoid errors during inpaint and extensions. 

**Please note that this is just a temporary solution and doesn't actually resolve the issue.It will try to use Euler method if error occurs.**

P.S.I trying to fix it……but all methods seems doesn't work.I've working for it over 36 hours.

**Suggestions from anyone are welcome**. 

I need to take a short break and prepare for my other project.*<== A mobile phone app base on flutter, using for TRPG.(No worries, I don't mean I will give up this project,also not about diverting traffic either. LOL.)*

想了个办法避免在局部重绘以及拓展中的报错。

**请注意，这只是一个临时解决方案，实际上并没有解决问题。如果出现错误，它将尝试使用欧拉方法。**

我努力尝试修复……但所有方案都不起作用。我已经连续工作了36小时以上。**欢迎任何人提出建议。**
，
我需要稍微休息一下，并且为我的其他项目做准备。*<== 一个基于Flutter的手机应用，用于TRPG。（别担心，我不是说我要放弃这个项目，也不是引流。）*

<a id="section8"></a>
## 2024.04.09

Add `__init__.py` for ComfyUI. Thanks for CapsAdmin. I don't use ComfyUI so I can't tell you how to add it, sorry.

为ComfyUI增加`__init__.py`  感谢CapsAdmin  我不用ComfyUI所以我没法告诉你怎么添加它，抱歉

<a id="section9"></a>
## Euler Smea Dyn Sampler

A sampling method based on Euler's approach, designed to generate superior imagery.

The SMEA sampler can significantly mitigate the structural and limb collapse that occurs when generating large images, and to a great extent, it can produce superior hand depictions (not perfect, but better than existing sampling methods).

The SMEA sampler is designed to accommodate the majority of image sizes, with particularly outstanding performance on larger images. It also supports the generation of images in unconventional sizes that lack sufficient training data (for example, running 512x512 in SDXL, 823x1216 in SD1.5, as well as 640x960, etc.).

The SMEA sampler performs very well in SD1.5, but the effects are not as pronounced in SDXL.

In terms of computational resource consumption, the Euler dy is approximately equivalent to the Euler a, while the Euler SMEA Dy sampler will consume more computational resources, approximately 1.25 times more.

一种基于Euler的采样方法，旨在生成更好的图片

Dyn采样器可以很大程度上避免出大图时的结构、肢体崩坏,能很大程度得到更优秀的手部（不完美但比已有采样方法更好）

Smea采样器理论上将增加图片的细节（**无法达到Nai3让图片闪闪发光的效果**）

适配绝大多数图片尺寸，在大图的效果尤其优秀，支持缺乏训练的异种尺寸（例如在sdxl跑512x512,在sd1.5跑823x1216,以及640x960等）

在SD1.5效果很好，在SDXL效果不明显。

计算资源消耗：Euler dy将约等于euler a, 而euler smea dy将消耗更多计算资源（约1.25倍）

<a id="section10"></a>
## Effect
**SD1.5，测试模型AnythingV5-Prt-RE，测试姿势Heart Hand,一个容易出坏手的姿势**

**SD1.5: Testing the AnythingV5-Prt-RE model with the Heart Hand pose often results in distorted hand positions.**

768x768,without Lora：
![xyz_grid-0049-1234-masterpiece,best quality,highres,1girl,heart hands,river,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/c13dcf8b-b9da-4624-9a04-a1488e850647)
768x768,with Lora：
![xyz_grid-0048-1234-_lora_manhattan_cafe_loha-000008_0 75_,manhattan cafe _(umamusume_),black choker,long sleeves,collared shirt,yellow necktie,blac](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/a60d7395-9ebd-4505-9ad7-3f0de0944bb8)
832x1216,without lora:
![xyz_grid-0046-1234-1girl,heart hands,river,cherry blossoms,hair flower,hair ribbon,cat ears,animal ear fluff,blue eyes,grey hair,short hair,bangs,h](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/9d6024f1-e9fa-42a0-841b-5a5b18089fd8)
832x1216,with Lora:
![xyz_grid-0047-1234-_lora_manhattan_cafe_loha-000008_0 75_,manhattan cafe _(umamusume_),black choker,long sleeves,collared shirt,yellow necktie,blac](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/3432dafb-4ebb-43ae-8c62-a4c4c168ce63)

**SDXL,测试模型animagineXLV31,测试姿势也是手部姿势**

**SDXL: Testing animagineXLV31 model with hand poses.**

768x768:
![xyz_grid-0019-114-1girl,manhattan cafe _(umamusume_),umamusume,heart hands,masterpiece,best quality,highres,](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/6883ace1-3712-4cd1-b974-4f0d1ed41bc2)
832x1216:
![xyz_grid-0018-114514-1girl,manhattan cafe _(umamusume_),umamusume,heart hands,masterpiece,best quality,highres,](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/9424ef70-a54f-454c-bf03-e543562367fc)
![xyz_grid-0019-114515-1girl,manhattan cafe _(umamusume_),umamusume,finger_on_trigger,upper body,masterpiece,best quality,highres,](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/2f9d1a68-3a28-47e5-a1a8-ce14e2ad4563)

<a id="section11"></a>
## how to use(This has become outdated, but it will be retained)

**step.1:** 打开`sd-webui-aki-v4.6\repositories\k-diffusion\k_diffusion`文件夹，打开其中的`sampling.py`文件（可以用记事本打开，称为文件1）

**Step 1:** Navigate to the `k_diffusion` folder within the `sd-webui-aki-v4.6\repositories\k-diffusion` directory and open the `sampling.py` file within it (this can be done using a text editor like Notepad, which will be referred to as File 1).
![QQ截图20240408193751](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/482d9647-f6d7-458e-a680-3c5a0c5fad9c)

**step.2:** 复制本仓库中的`sampling.py`中的所有内容并粘贴到文件1末尾

**Step 2:** Copy the entire content from the `sampling.py` file in the current repository and paste it at the end of File 1.
![image](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/2b48db77-2d44-410e-afe7-81f04ede34ce)
(To present the complete picture, I have utilized PyTorch's abbreviation feature.)

**Step 3:** Open the `sd_samplers_kdiffusion.py` file located in the `sd-webui-aki-v4.6\modules` directory (refer to this as File 2).
![image](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/e014a51f-7d8a-49ee-ba3d-ec5d338f51d1)

**Step 4:** Copy the following two lines from this repository:
![QQ截图20240408192923](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/440f522a-1acb-4d78-bebe-b75c7b969adb)

Paste them into File 2:
![image](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/782722bf-713c-41dc-a7ec-669419423ae5)

**Step 5:** Restart the webui, and you will see:
![image](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler/assets/66173435/15abbf9e-4d4d-4325-9223-1506f08f25cc)

现在你就可以使用它们了。在图生图中可能有一些bug，欢迎向我汇报（请带上截图/报错声明）

Now you can start using them. There may be some bugs in the image generation process, and I welcome you to report any issues to me (please provide screenshots or error statements).

<a id="section12"></a>
## The technical principles

简单地讲，dyn方法有规律地取出图片中的一部分，去噪后加回原图。在理论上这应当等同于euler a，但其加噪环节被替代为有引导的噪声。

而smea方法将图片潜空间放大再压缩回原本的大小，这增加了图片的可能性。很抱歉我没能实现Nai3中smea让图片微微发光的效果。

一点忠告：不要相信pytorch的插值放大和缩小方法，不会对改善图像带来任何帮助。同时用有条件引导取代随机噪声也是有希望的道路。

In simple terms, the dyn method regularly extracts a portion of the image, denoises it, and then adds it back to the original image. Theoretically, this should be equivalent to the Euler A method, but its noise addition step is replaced with guided noise.

The SMEA method enlarges the image's latent space and then compresses it back to its original dimensions, thereby increasing the range of possible image variations. I apologize that I was unable to achieve the subtle glowing effect in Nai3 with the SMEA method.

A piece of advice: Do not trust PyTorch's interpolation methods for enlarging and shrinking images; they will not contribute to improving image quality. Additionally, replacing random noise with conditional guidance is also a promising path forward.

<a id="section13"></a>
## Contact the author

Email:872324454@qq.com

Bilibili:星河主炮发射
