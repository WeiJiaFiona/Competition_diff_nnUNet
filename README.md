## Welcome to **Universal Segmentation of HCC Using DDPM and nnU-Net** ;-)<br>
##### 🏆获奖情况： **全国一等奖（TOP 5%)** | 2024第九届全国大学生生物医学工程创新设计竞赛<br>
##### 👥 项目成员： [贾薇](https://github.com/WeiJiaFiona)（组长）\,吴锦泽,  陈咏茜, 牟悠然, 朱宸磊
##### 📚指导老师：[崔智铭](https://shanghaitech-impact.github.io/)<sup>+</sup>，[张增](https://bme.shanghaitech.edu.cn/2021/0326/c8204a1080187/page.htm)<sup>+</sup><br>
##### 🙏特别感谢：[吴瀚](https://hanwu.website/)的支持<br>

***
#### (1) 比赛简介<br>
* 2024第九届全国大学生生物医学工程创新设计竞赛
* **中国生物医学学会**主办
* 全国本硕博**2000**+队伍海南大学线下参赛
* 获奖比例：一等奖（5%），二等奖（15%），三等奖（25%）<br>


#### (2) 赛题背景<br>
* 在赛方给定的肝细胞癌患者的静脉期腹部CT中可能存在多个肿瘤，但仅提供最大肿瘤的标注，测试时，要求分割出每例CT中的最大肿瘤并提交官方计算DSC与HD95等指标进行排名
* 问题定义：
    * 部分标注导致的肿瘤表征不一致问题
    * 部分患者处于肝细胞癌早期，小病灶难以检测，且此类数据较少<br>
#### (3) 项目创新<br>
* **半监督学习**：补全未标注肿瘤的pseudo label，增强标注一致性，允许更全面准确的表征学习
* **以mask为条件使用DDPM引导生成**更多的小肿瘤数据
* **数据分析**：使用统计学方法全面理解数据集，例如直径，面积，球型度等<br>

#### (4) 代码复现<br>
##### 1. 比赛官方数据集下载地址
训练集 链接： https://pan.baidu.com/s/1WxeMniaDTNdpZVEY-GUfgA
提取码： w2ux
外部测试集 链接： https://pan.baidu.com/s/1X4rlAz5JL5MEE8t4_FKaLQ
提取码： oe1f<br>
##### 2. 我们本次比赛训练的权重<br>
半监督补全nnUNet权重 链接： https://pan.baidu.com/s/1gXjOJtfCVSf2qZ8frir0Nw?pwd=cuc3
提取码： cuc3
最终用于预测测试集的Diff-nnUNet权重 链接：https://pan.baidu.com/s/18RxT0ncf44yG6xRHsLU4LA?pwd=3bfq
提取码： oe1f<br>
##### 3. 训练你的模型<br>
* **肿瘤数据集统计分析**： 
`tumor_statistics`文件夹下含肿瘤物理直径、体积、球型度的统计
* **Step1 半监督补全标签** ： 
运行步骤详见`Step1_nnUNet_pseudo_label\pseudo_label_README.md`  
* **Step2 预测肝脏标签**： 
运行步骤详见`Step2_CLIP-Driven-Universal-Model\README.md`
* **Step3 DDPM合成小肿瘤数据及训练最终预测模型Diff-nnUNet**： 
运行步骤详见`Step3_Diff_nnUNet\README.md`<br>
#### (5) 测试结果<br>
##### 1. 外部测试集结果
* 在同赛道的36支队伍中**排名第一**<br>
##### 2. 内部测试集结果<br>
* **内部测试集划分**：为了初步验证模型效果，我们从训练集中额外划分出内部测试集。将官方训练集肿瘤按照物理体积从小到大排列后，等间距抽样出20个sample，能够涵盖从极小到极大的肿瘤，测试具有一定的代表性<br>
* **内部测试集结果**：`Internal Test Set_DSC_HD`文件夹下包含统计HD95及DSC指标的代码与结果<br>

* **可视化与统计展示**：<br>
<img src="imgs\internal_test_statitics.png" alt="Internal Test Statistics" width="500"><br>
<img src="imgs\vis1.png" alt="vis1" width="500"><br>
<img src="imgs\vis2.png" alt="vis1" width="500"><br>
<!-- ![vis1](imgs\vis1.png)
![vis2](imgs\vis2.png) -->
#### (6) 项目总结
本项目通过补全标注与合成数据提升了数据的质量与数量，进行消融实验验证了核心设计的有效性，并在外部测试集上获得优异表现，再次说明了**Deep learning is data-driven**的底层逻辑。