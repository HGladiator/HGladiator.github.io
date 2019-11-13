---
layout: post
title: 一个框架解决几乎所有机器学习问题
categories: AI中台
tags: AutoML
excerpt: 上周一个叫 Abhishek Thakur 的数据科学家，在他的 Linkedin 发表了一篇文章 [Approaching (Almost) Any Machine Learning Problem](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/)，介绍他建立的一个自动的机器学习框架，几乎可以解决任何机器学习问题，项目很快也会发布出来。这篇文章迅速火遍 Kaggle，他参加过100多个数据科学相关的竞赛，积累了很多宝贵的经验，看他很幽默地说“写这样的框架需要很多丰富的经验，不是每个人都有这样的经历，而很多人有宝贵的经验，但是他们不愿意分享，我呢恰好是又有一些经验，又愿意分享的人”。当然这篇文章也是受到争议的，很多人觉得并不全面。
mathjax: true
date: 2019-03-26
author:
---



* content
{:toc}




我最近也在准备参加 Kaggle，之前看过几个例子，自己也总结了一个分析的流程，今天看了这篇文章，里面提到了一些高效的方法，最干货的是，他做了一个表格，列出了各个算法通常需要训练的参数。

这个问题很重要，因为大部分时间都是通过调节参数，训练模型来提高精度。作为一个初学者，第一阶段，最想知道的问题，就是如何调节参数。因为分析的套路很简单，就那么几步，常用的算法也就那么几个，以为把算法调用一下就可以了么，那是肯定不行的。实际过程中，调用完算法后，结果一般都不怎么好，这个时候还需要进一步分析，哪些参数可以调优，哪些数据需要进一步处理，还有什么更合适的算法等等问题。

## **接下来一起来看一下他的框架。**

据说数据科学家 60-70％ 的时间都花在数据清洗和应用模型算法上面，这个框架主要针对算法的应用部分。

![Pipeline](https://blog-10039692.file.myqcloud.com/1507706111793_5442_1507706107871.png)


## 什么是 Kaggle？

Kaggle是一个数据科学竞赛的平台，很多公司会发布一些接近真实业务的问题，吸引爱好数据科学的人来一起解决，可以通过这些数据积累经验，提高机器学习的水平。

应用算法解决 Kaggle 问题，一般有以下几个步骤：

*   第一步：识别问题
*   第二步：分离数据
*   第三步：构造提取特征
*   第四步：组合数据
*   第五步：分解
*   第六步：选择特征
*   第七步：选择算法进行训练

当然，工欲善其事，必先利其器，要先把工具和包都安好。 最方便的就是安装 Anaconda，这里面包含大部分数据科学所需要的包，直接引入就可以了，常用的包有：

*   pandas：常用来将数据转化成 dataframe 形式进行操作
*   scikit-learn：里面有要用到的机器学习算法模型
*   matplotlib：用来画图
*   以及 xgboost，keras，tqdm 等。

### 第一步：确定问题

在这一步先明确这个问题是分类还是回归。通过问题和数据就可以判断出来，数据由 X 和 label 列构成，label 可以一列也可以多列，可以是二进制也可以是实数，当它为二进制时，问题属于分类，当它为实数时，问题属于回归。

### 第二步：分离数据

![](https://blog-10039692.file.myqcloud.com/1507706161329_2449_1507706157270.png)


用 Training Data 来训练模型，用 Validation Data 来检验这个模型的表现，不然的话，通过各种调节参数，模型可以在训练数据集上面表现的非常出色，但是这可能会是过拟合，过拟合就是太依赖现有的数据了，拟合的效果特别好，但是只适用于训练集，以致于来一个新的数据，就不知道该预测成什么了。所以需要有 Validation 来验证一下，看这个模型是在那里自娱自乐呢，还是真的表现出色。

在 scikit learn 包里就有工具可以帮你做到这些： 

* 分类问题用 StrtifiedKFold
![](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_4.png?w=675)

* 回归问题用 KFold，但是，有一些复杂的方法往往会使训练和验证集的标签分布保持相同，这留给读者练习。
![](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_5.png?w=671)

### 第三步：构造特征

这个时候，需要将数据转化成模型需要的形式。数据有三种类型：数字，类别，文字。首先分离出数值变量。这些变量不需要任何处理，因此我们可以开始将规范化和机器学习模型应用于这些变量。
我们可以通过两种方式处理分类数据，这个过程 sklearn 也可以帮你做到：
*  将分类数据转换为标签	

![abhishek_7](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_7.png?resize=800%2C173)

* 将标签转换为二进制变量（单编码）

![abhishek_8](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_8.png?resize=800%2C177)

请记住在将OneHotEncoder应用于类别之前，先使用LabelEncoder将类别转换为数字。

### 第四步：组合数据
由于Titanic数据没有很好的文本变量示例，因此让我们制定一个处理文本变量的一般规则。我们可以将所有文本变量组合为一个，然后使用一些对文本数据起作用并将其转换为数字的算法。

文本变量可以按如下方式连接：

![abhishek_9](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_9.png?resize=800%2C30)

然后，我们可以在其上使用CountVectorizer或TfidfVectorizer：

![abhishek_10](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_10.png?resize=766%2C111)

要么，

![abhishek_11](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_11.png?resize=755%2C113)

在大多数情况下，TfidfVectorizer的性能要好于计数，并且我已经看到，TfidfVectorizer的以下参数几乎始终有效。

![abhishek_12](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_12.png?resize=800%2C166)

如果仅将这些矢量化器应用于训练集，请确保将其转储到硬盘驱动器，以便以后可以在验证集中使用它。

![abhishek_13](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_13.png?resize=741%2C62)

接下来，我们来介绍堆栈器模块。堆栈器模块不是模型堆栈器，而是功能堆栈器。可以使用堆栈器模块组合上述处理步骤后的不同功能。

![abhishek_14](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_14.png?resize=611%2C134)

您可以先水平堆叠所有功能，然后再使用numpy hstack或sparse hstack对其进行进一步处理，具体取决于您具有密集功能还是稀疏功能。

![abhishek_15](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_15.png?resize=400%2C243)

如果还有其他处理步骤（例如pca或功能选择），也可以通过FeatureUnion模块来实现（我们将在本文后面访问分解和功能选择）。

![abhishek_16](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_16.png?resize=800%2C228)

一旦将功能堆叠在一起，就可以开始应用机器学习模型。在此阶段，仅应使用的模型应该是基于集成树的模型。这些模型包括

组合之后，就可以应用以下算法模型：

*   RandomForestClassifier
*   RandomForestRegressor
*   ExtraTreesClassifier
*   ExtraTreesRegressor
*   XGBClassifier
*   XGBRegressor

但是不能应用线性模型，线性模型之前需要对数据进行正则化而不是上述预处理。

### 第五步：分解

这一步是为了进一步优化模型，可以用以下方法：

![](https://blog-10039692.file.myqcloud.com/1507706259451_2054_1507706255470.png)

PCA：Principal components analysis，主成分分析，是一种分析、简化数据集的技术。用于减少数据集的维数，同时保持数据集中的对方差贡献最大的特征。

为了简单起见，我们将省略LDA和QDA转换。对于高维数据，通常使用PCA分解数据。对于图像，从10到15个分量开始，并增加此数目，只要结果质量得到显着改善。对于其他类型的数据，我们最初选择50-60个分量（只要我们可以按原样处理数值数据，就倾向于避免使用PCA）。

![abhishek_18](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_18.png?resize=400%2C134)

对于文本数据，将文本转换为稀疏矩阵后，请进行奇异值分解（SVD）。可以在scikit-learn中找到称为TruncatedSVD的SVD变体。
**SVD：Singular Value Decomposition，奇异值分解**，是线性代数中一种重要的矩阵分解，它总能找到标准化正交基后方差最大的维度，因此用它进行降维去噪。

![abhishek_decomp](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_decomp.png?resize=681%2C180)

通常可用于TF-IDF的SVD组件数或计数在120-200之间。高于此数目的任何数字都可能会提高性能，但不会显着改善，并且以计算能力为代价。

在评估了模型的进一步性能之后，我们转向数据集的缩放，以便我们也可以评估线性模型。然后可以将标准化或按比例缩放的特征发送到机器学习模型或特征选择模块。

![abhishek_19](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_19.png?resize=543%2C156)




### 第六步：选择特征

有多种方法可以实现特征选择。最常见的方法之一是贪婪特征选择（向前或向后）。在贪婪特征选择中，我们选择一个特征，训练模型并以固定的评估指标评估模型的性能。我们会一一保持添加和删除功能，并记录每一步模型的性能。然后，我们选择具有最高评分的功能。可以在这里找到以AUC作为评估指标的贪婪特征选择的一种实现：[greedyFeatureSelection](https://github.com/abhishekkrthakur/greedyFeatureSelection) 。必须注意的是，这种实现方式不是完美的，必须根据要求进行更改/修改。

其他更快的特征选择方法包括从模型中选择最佳特征。我们可以查看logit模型的系数，也可以训练随机森林以选择最佳特征，然后在其他机器学习模型中使用它们。

![abhishek_20](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_20.png?resize=800%2C177)

请记住，要保持较少的估计量，并尽量减少对超参数的优化，以免过大。

特征选择也可以使用渐变增强机来实现。最好在scikit-learn中使用xgboost代替GBM的实现，因为xgboost更快，更具伸缩性。

![abhishek_21](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_21.png?resize=800%2C213)

我们还可以使用RandomForestClassifier / RandomForestRegressor和xgboost进行稀疏数据集的特征选择。

从正稀疏数据集中进行特征选择的另一种流行方法是基于chi-2的特征选择，我们也已在scikit-learn中实现了该方法。

![abhishek_22](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_22.png?resize=722%2C183)

在这里，我们将chi2与SelectKBest结合使用以从数据中选择20个特征。这也成为我们要优化以改善机器学习模型结果的超参数。

不要忘记在所有步骤中使用的任何类型的变压器。您将需要它们来评估验证集上的性能。

下一步（或中间步骤）的主要步骤是模型选择+超参数优化。

![abhishek_23](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2016/07/abhishek_23.png?resize=387%2C400)
当特征个数越多时，分析特征、训练模型所需的时间就越长，容易引起“维度灾难”，模型也会越复杂，推广能力也会下降，所以需要剔除不相关或亢余的特征。

常用的算法有完全搜索，启发式搜索，和随机算法。
```
#例如，Random Forest：

from sklearn.ensemble import RandomForestClassifier

#或者 xgboost：

import xgboost as xgb

#对于稀疏的数据，一个比较有名的方法是 chi-2：

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
```
### 第七步：选择算法进行训练

选择完最相关的参数之后，接下来就可以应用算法，常用的算法有：

*   **Classification**:
	*   Random Forest
	*   GBM
	*   Logistic Regression
	*   Naive Bayes
	*   Support Vector Machines
	*   k-Nearest Neighbors

*   **Regression**
	*   Random Forest
	*   GBM
	*   Linear Regression
	*   Ridge
	*   Lasso
	*   SVR


在[scikit－learn](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)里可以看到分类和回归的可用的算法一览，包括它们的原理和例子代码。

在应用各算法之前先要明确这个方法到底是否合适。 为什么那么多算法里，只提出这几个算法呢，这就需要对比不同算法的性能了。 这篇神文 [Do we Need Hundreds of Classifiers to Solve Real World Classification Problems](http://jmlr.org/papers/v15/delgado14a.html) 测试了179种分类模型在UCI所有的121个数据上的性能，发现Random Forests 和 SVM 性能最好。 我们可以学习一下里面的调研思路，看看是怎么样得到比较结果的，在我们的实践中也有一定的指导作用。

![](https://blog-10039692.file.myqcloud.com/1507706339459_6332_1507706335760.png)

各算法比较

但是直接应用算法后，一般精度都不是很理想，这个时候需要调节参数，最干货的问题来了，什么模型需要调节什么参数呢？

![](https://blog-10039692.file.myqcloud.com/1507706352167_4129_1507706348227.png)

虽然在sklearn的文档里，会列出所有算法所带有的参数，但是里面并不会说调节哪个会有效。在一些mooc课程里，有一些项目的代码，里面可以看到一些算法应用时，他们重点调节的参数，但是有的也不会说清楚为什么不调节别的。这里作者根据他100多次比赛的经验，列出了这个表，我觉得可以借鉴一下，当然，如果有时间的话，去对照文档里的参数列表，再查一下算法的原理，通过理论也是可以判断出来哪个参数影响比较大的。

调参之后，也并不就是大功告成，这个时候还是需要去思考，是什么原因造成精度低的，是哪些数据的深意还没有被挖掘到，这个时候需要用**统计和可视化去再一次探索数据**，之后就再走一遍上面的过程。

我觉得这里还提到了很有用的一条经验是，**把所有的 transformer 都保存起来**，方便在 validation 数据集上面应用：

![](https://blog-10039692.file.myqcloud.com/1507706374144_6119_1507706370146.png)

文章里介绍了分析问题的思路，还提到了几条很实用的经验，不过经验终究是别人的经验，只能借鉴，要想提高自己的水平，还是要看到作者背后的事情，就是参加了100多次实战，接下来就去行动吧，享受用算法和代码与数据玩耍的兴奋吧。