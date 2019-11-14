---
layout: post
title: Hopsworks系统中的FeatureStore详解
categories: AI中台
tags: FeatureStore  AutoML
excerpt: 构建成功的AI系统非常困难。在Logical Clocks，我们注意到我们的用户在机器学习的数据工程阶段花费了大量精力。从版本0.8.0开始，Hopsworks提供了世界上第一个开源FeatureStore。FeatureStore是用于机器学习的数据管理层，它允许数据科学家和数据工程师共享和发现特征，随着时间的推移更好地了解特征，并实现机器学习工作流程。
mathjax: false
date: 2019-04-10
---



* content
{:toc}

# Feature Store：ML pipelines中缺少数据层吗？

> *   原文地址：[# Feature Store: The missing data layer in ML pipelines?](https://www.logicalclocks.com/blog/feature-store-the-missing-data-layer-in-ml-pipelines)
> *   原文作者：Kim Hammar & Jim Dowling


## 使用FeatureStore把ML pipelines模块化

TLDR；FeatureStore是一个中央保管库，用于存储文档化，策展和访问控制的特征。在此博客文章中，我们讨论了用于深度学习的数据管理的最新技术，并展示了Hopsworks中提供的第一个开源FeatureStore。

## 什么是Feature Store？

FeatureStore的概念[由Uber在2017年提出[11]](https://www.google.com/url?q=https://eng.uber.com/michelangelo/&sa=D&ust=1546167503203000)。特征库是在组织内存储精选特征的中心位置。特征是某些数据样本的可测量属性。例如，它可能是图像像素，一段文字中的单词，一个人的年龄，从传感器发出的坐标或一个汇总值，例如最近一小时内的平均购买次数。可以直接从文件和数据库表中提取要素，也可以从一个或多个数据源中计算出派生值。

特征是AI系统的动力，因为我们使用它们来训练机器学习模型，以便我们可以预测以前从未见过的特征值。

![图1.特征库是特征工程与模型开发之间的接口](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7b20766201869af7cf0f_feature-store-blog-image.png)


### FeatureStore具有两个接口：

**写入FeatureStore区：**数据工程师的界面。在要素工程流程的最后，不是将要素写入文件或特定于项目的数据库或文件，而是将要素写入要素存储。

###### 数据工程师示例：
```py
from hops import featurestore
raw_data = spark.read.parquet(filename)

polynomial_features = raw_data.map(lambda x: x^2)

featurestore.insert_into_featuregroup(polynomial_features, "polynomial_featuregroup")
```
**从特征部件存储中读取：**数据科学家接口。要在一组要素上训练模型，可以直接从要素存储中读取要素。

###### 数据科学家示例：
```py
from hops import featurestore
‍
features_df = featurestore.get_features(["average_attendance", "average_player_age"])
```
要素存储不是简单的数据存储服务，它还是数据转换服务，因为它使要素工程成为一流的构造。特征工程是将原始数据转换为预测模型可以理解的格式的过程。

## 为什么需要FeatureStore

在Logical Clocks，我们致力于开发技术来大规模运行机器学习工作流程，并帮助组织从数据中提取情报。机器学习是一种非常强大的方法，它有可能帮助我们从对世界的历史了解转变为对我们周围世界的预测模型。但是，构建机器学习系统非常困难，并且需要专门的平台和工具。

尽管临时特征工程和培训管道是数据科学家尝试机器学习模型的快速方法，但随着时间的流逝，这种管道有变得复杂的趋势。随着模型数量的增加，它很快成为难以管理的管道丛林。这激励了在特征工程过程中使用标准化方法和工具，从而有助于降低开发新的预测模型的成本。FeatureStore是为此目的而设计的服务。

### 机器学习系统中的技术债务

> “机器学习：技术债务的高利息信用卡”

– [谷歌[3]](https://www.google.com/url?q=https://static.googleusercontent.com/media/research.google.com/sv//pubs/archive/43146.pdf&sa=D&ust=1546167503207000)

机器学习系统倾向于聚集技术债务[[1]](https://www.google.com/url?q=https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf&sa=D&ust=1546167503207000)。机器学习系统中的技术债务示例包括：

*   在模型服务期间，没有原则性的方式来访问要素。
*   无法轻松地在多个机器学习管道之间重用特征。
*   数据科学项目是孤立工作的，无需协作和重复使用。
*   用于培训和服务的特征不一致。
*   当有新数据到达时，无法准确确定需要重新计算哪些特征，而是需要运行整个管道来更新特征。

由于技术的复杂性，我们已经说过的几个组织在努力扩展其机器学习工作流时，考虑到它的高技术成本，有些团队甚至不愿采用机器学习。使用FeatureStore是一种最佳实践，可以减少机器学习工作流的技术负担。

> “只有从整体上考虑数据收集和特征提取，才能避免流水线丛林”

– [谷歌[1]](https://www.google.com/url?q=https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf&sa=D&ust=1546167503208000)

## 数据工程是机器学习中最难的问题

> 数据是ML中最难的部分，也是正确的最重要的部分。建模人员花费大量时间在培训时选择和转换特征，然后构建将这些特征传递到生产模型的管道。数据损坏是生产ML系统中出现问题的最常见原因”

– [优步[2]](https://www.google.com/url?q=https://eng.uber.com/scaling-michelangelo/&sa=D&ust=1546167503208000)

在生产中和大规模交付机器学习解决方案与将模型拟合到预处理数据集有很大不同。实际上，开发模型的大部分工作都花在了特征工程和数据整理上。

![图2.模型开发只是机器学习项目工作的一部分](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7c287662010251f7d1f7_image5.png)


从原始数据提取特征的方法有很多，但是常见的特征工程步骤包括：

*   将分类数据转换为数字数据；
*   归一化数据（以减轻特征源自不同分布时的不良条件优化）；
*   一热编码/二值化；
*   特征合并（例如，将连续特征转换为离散特征）；
*   特征哈希（例如，减少一键编码特征的内存占用）；
*   计算多项式特征；
*   表征学习（例如，使用聚类，嵌入或生成模型来提取特征）；
*   计算集合特征（例如，count，min，max，stdev）。

为了说明要素工程的重要性，让我们考虑对只有一个要素_x1_的数据集进行分类任务，如下所示：

![图3.具有单个特征_x1的_数据集，具有两个不可线性分离的类（实心圆和非实心圆）](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7c2989835660dbc7dab5_image13.png)



如果我们尝试将线性模型直接拟合到此数据集，则注定失败，因为它不是线性可分离的。在特征工程期间，我们可以提取一个附加特征_x2_，其中从原始数据集中派生_x2_的函数为_x2 __=（x1）^ 2_。所得的二维数据集可能如图2所示。

![图4.具有两个特征_x1_和_x2的_数据集，具有两个可线性分离（例如，用红线表示）的类（实心圆和非实心圆)](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7c297662013462f7d1f9_image19.png)


通过添加额外的特征，数据集可以线性分离，并且可以由我们的模型拟合。这是一个简单的示例，实际上，特征工程的过程可能涉及更复杂的转换。

在深度学习的情况下，[深度模型往往会](https://www.google.com/url?q=https://blog.acolyer.org/2018/03/28/deep-learning-scaling-is-predictable-empirically/&sa=D&ust=1546167503214000)在训练的数据[越多](https://www.google.com/url?q=https://blog.acolyer.org/2018/03/28/deep-learning-scaling-is-predictable-empirically/&sa=D&ust=1546167503214000)的情况下[发挥更好的性能](https://www.google.com/url?q=https://blog.acolyer.org/2018/03/28/deep-learning-scaling-is-predictable-empirically/&sa=D&ust=1546167503214000)（训练过程中的更多数据样本会产生正则化效果并打击过度拟合）。因此，机器学习的趋势是在越来越大的数据集上进行训练。这种趋势使特征工程流程更加复杂，因为数据工程师除特征工程逻辑外还必须考虑可伸缩性和效率。使用标准化且可扩展的特征平台，可以更有效地管理特征工程的复杂性。

## 没有FeatureStore之前的工作流程

![图5.没有FeatureStore的典型机器学习基础结构](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7e5d8f10c52ef8ff357f_image9.png)


### 没有用到FeatureStore的时候

在图5中，特征代码在培训工作中重复，并且还有一些特征具有不同的实现：一种用于培训，另一种用于部署（推理）（模型C）。具有用于训练和部署的计算特征的不同实现方式需要非DRY代码，并且可能导致预测问题。此外，如果没有FeatureStore，特征通常无法重用，因为它们已嵌入培训/服务工作中。这也意味着数据科学家必须编写底层代码才能访问数据存储，这需要数据工程技能。也没有搜索特征实现的服务，也没有特征的管理或治理。

![图6.具有FeatureStore的机器学习基础结构](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7e5d898356cd7dc7e6ca_image14.png)



### 使用FeatureStore之后

数据科学家现在可以搜索特征，并借助API支持，以最少的数据工程轻松地使用它们来构建模型。此外，其他模型可以缓存和重用特征，从而减少了模型训练时间和基础架构成本。特征现在是企业中的一项托管资产。

## 机器学习组织的规模经济

对于应用机器学习的组织来说，经常遇到的一个陷阱是将数据科学团队视为独立的团队，它们之间的协作有限。具有这种思维方式会导致机器学习工作流，其中没有标准化的方法可以在不同团队和机器学习模型之间共享特征。无法在模型和团队之间共享特征限制了数据科学家的工作效率，并使得构建新模型更加困难。通过使用共享FeatureStore，组织可以实现**规模经济**效应。当特征部件库具有更多特征部件时，**构建新模型变得更加容易和便宜，因为新模型可以重复使用特征部件库中存在的特征部件。**

![图7.通过在组织内集中存储特征，缩短了新模型和机器学习项目的启动周期](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e7e60240e831f05a83b97_image16.png)



‍

## HopsworksFeatureStore

随着Hopsworks 0.8.0的发布，我们将发布第一个开源FeatureStore服务，该服务将集成到[HopsML框架中[8]](https://www.google.com/url?q=https://hops.readthedocs.io/en/latest/hopsml/hopsML.html&sa=D&ust=1546167503216000)。在本节中，我们将介绍系统的技术细节及其使用方法。

### 要素存储的组成部分和现有要素存储的比较

在2018年期间，许多在大规模应用机器学习方面处于领先地位的大公司宣布开发专有FeatureStore。Uber，LinkedIn和Airbnb在Hadoop数据湖上构建了FeatureStore，而Comcast在AWS数据湖上构建了FeatureStore，而GO-JEK在Google数据平台上构建了FeatureStore。

这些现有FeatureStore由五个主要组件组成：

*   **特征工程工作**，特征计算，特征计算的主要框架是Samza（[Uber [4]](https://www.google.com/url?q=https://static.googleusercontent.com/media/research.google.com/sv//pubs/archive/43146.pdf&sa=D&ust=1546167503219000)），Spark（[Uber [4]](https://www.google.com/url?q=https://static.googleusercontent.com/media/research.google.com/sv//pubs/archive/43146.pdf&sa=D&ust=1546167503219000)，[Airbnb [5]](https://www.google.com/url?q=https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform&sa=D&ust=1546167503219000)，[Comcast [6]](https://www.google.com/url?q=https://databricks.com/session/operationalizing-machine-learning-managing-provenance-from-raw-data-to-predictions&sa=D&ust=1546167503220000)），Flink （[Airbnb [5]](https://www.google.com/url?q=https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform&sa=D&ust=1546167503220000)，[Comcast [6]](https://www.google.com/url?q=https://databricks.com/session/operationalizing-machine-learning-managing-provenance-from-raw-data-to-predictions&sa=D&ust=1546167503220000)）和Beam（[GO-JEK [7]](https://www.google.com/url?q=https://www.youtube.com/watch?v%3D0iCXY6VnpCc&sa=D&ust=1546167503221000)）。
*   用于存储要素数据**的存储层**。存储特征的常见解决方案是Hive（[Uber [4]](https://www.google.com/url?q=https://static.googleusercontent.com/media/research.google.com/sv//pubs/archive/43146.pdf&sa=D&ust=1546167503221000)，[Airbnb [5]](https://www.google.com/url?q=https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform&sa=D&ust=1546167503221000)），S3（[Comcast [6]](https://www.google.com/url?q=https://databricks.com/session/operationalizing-machine-learning-managing-provenance-from-raw-data-to-predictions&sa=D&ust=1546167503221000)）和BigQuery（[GO-JEK [7]](https://www.google.com/url?q=https://www.youtube.com/watch?v%3D0iCXY6VnpCc&sa=D&ust=1546167503221000)）。
*   用于存储代码以计算要素，要素版本信息，要素分析数据和要素文档**的元数据层**。
*   **Feature Store API，**用于从要素存储中读取要素或向要素存储中写入要素。
*   **特征注册表**，数据科学家可以在其中共享，发现和订购特征计算的用户界面（UI）服务。

在深入探讨FeatureStore区API及其用法之前，让我们看一下构建FeatureStore区所基于的技术堆栈。

### Hopsworks  FeatureStore架构

FeatureStore的架构如图8所示。

![图8\. Hopsworks Feature Store的体系结构](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e83be89835659c8c80829_image18.png)



### 特征工程框架

在Logical Clocks，我们专门研究Python优先的ML管道，对于特征工程，我们将支持重点放在Spark，PySpark，Numpy和Pandas。使用Spark / PySpark进行特征工程的动机是，它是使用大型数据集的用户之间进行数据争吵的首选。但是，我们还观察到，使用小型数据集的用户更喜欢使用诸如Numpy和Pandas之类的框架进行特征工程，这就是为什么我们决定也为这些框架提供本机支持的原因。用户可以使用笔记本，python文件或.jar文件在Hopsworks平台上提交要素工程作业。

### 储存层

我们在Hive / HopsFS之上构建了用于要素数据的存储层，并具有用于要素数据建模的其他抽象。使用Hive作为基础存储层的原因有两个：（1）我们的用户正在使用TB级或更大规模的数据集，并且要求可在HopsFS上部署可伸缩解决方案，这并不罕见[。 HopsFS [9]](https://www.google.com/url?q=https://www.logicalclocks.com/fixing-the-small-files-problem-in-hdfs/&sa=D&ust=1546167503223000)）; （2）自然地以关系方式完成要素的数据建模，将关系要素分组到表中并使用SQL查询要素存储。这种类型的数据建模和访问模式非常适合与Hive结合使用柱状存储格式（例如Parquet或ORC）。

### 元数据层

为了提供自动版本控制，文档编制，特征分析和特征共享，我们将有关特征的扩展元数据存储在元数据存储中。对于元数据存储，我们利用NDB（MySQL群集），它使我们能够保持要素元数据与Hopsworks中的其他元数据高度一致，例如有关要素工程作业和数据集的元数据。

### 特征数据建模

我们向用户介绍了三个新概念，用于对要素存储中的数据进行建模。

*   **特征**是FeatureStore中的单独版本化和记录的数据列，例如，客户的平均评分。
*   **特征组**是存储为Hive表的文档化和版本控制的特征组。要素组链接到特定的Spark / Numpy / Pandas作业，该作业接收原始数据并输出计算出的要素。
*   **训练数据集**是特征和标签的版本控制的数据集（可能来自多个不同的特征组）。训练数据集作为tfrecords，parquet，csv，tsv，hdf5或.npy文件存储在HopsFS中。

![图9.特征组包含一组特征，而训练数据集包含一组特征，可能来自许多不同的特征组](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e83bf240e836d36a85953_image8.png)


设计要素组时，最好的做法是让从同一原始数据集计算出的所有要素都位于同一要素组中。常见的是，有多个要素组共享一个公共列，例如时间戳或客户ID，这些要素组可将要素组结合在一起形成训练数据集。

## Feature Store API

特征部件存储具有两个接口。一个界面用于将选定的特征写入特征存储，一个界面用于从特征存储中读取特征以用于训练或服务。  

### 创建特征

特征存储与用于计算特征的方法无关。唯一的要求是可以将这些特征分组到Pandas，Numpy或Spark数据框中。用户为数据框提供要素和关联的要素元数据（也可以稍后通过要素注册表UI编辑元数据），要素存储库负责创建要素组的新版本，计算要素统计信息并将要素链接到计算它们的工作。

###### 插入特征
```py
from hops import featurestore
featurestore.insert_into_featuregroup(features_df, featuregroup_name)
```
###### Create Feature Group
```py
from hops import featurestore

featurestore.create_featuregroup(
   features_df,
   featuregroup_name,
   featuregroup_description,
   feature_engineering_job,
   featuregroup_version
)
```
### 从特征部件存储中读取（查询计划器）

要从FeatureStore中读取特征，用户可以在Python和Scala中使用SQL或API。根据我们在平台上与用户的经验，数据科学家可以具有不同的背景。尽管一些数据科学家对SQL非常满意，但其他一些科学家则更喜欢高级API。这促使我们开发了一个查询计划器来简化用户查询。查询计划程序使用户能够表达最基本的信息，以便从要素存储中获取要素。例如，用户只需提供特征名称列表，就可以请求分布在20个不同特征组中的100个特征。查询计划人员使用要素存储中的元数据来推断从何处获取要素以及如何将要素结合在一起。

![图10.用户以编程方式或使用SQL查询来查询FeatureStore。输出以Pandas，Numpy或Spark数据帧形式提供](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e84e6aa3dfe7d62a6dac6_image17.png)



要从特征部件存储中获取特征“ average_attendance”和“ average_player_age”，用户只需编写即可。

###### 示例获取特征
```py
from hops import featurestore
features_df = featurestore.get_features(["average_attendance", "average_player_age"])
```
返回的“ features_df” _是一个（Pandas，Numpy或Spark）数据框，然后可用于生成模型的训练数据集。_

### 创建训练数据集

组织通常具有许多不同类型的原始数据集，可用于提取要素。例如，在用户推荐的情况下，可能存在一个包含用户人口统计数据的数据集，另一个包含用户活动的数据集。来自同一数据集的要素自然会分为一个**要素组**，通常每个数据集生成一个要素组。训练模型时，您要包括对预测任务具有预测能力的所有要素，这些要素可能跨越多个要素组。为此，使用了Hopsworks Feature Store中的训练数据集抽象。训练数据集允许用户使用标签对一组特征进行分组，以训练模型执行特定的预测任务。

一旦用户从要素存储中的不同要素组中获取了一组要素，就可以将要素与标签结合在一起（在监督学习的情况下），然后将其具体化为训练数据集。通过使用特征部件存储API创建训练数据集，该数据集将由特征部件存储进行_管理_。托管的训练数据集会自动进行数据异常分析，版本化，记录和与组织共享。

![图11\. HopsML中的数据生命周期。原始数据被转换为特征，这些特征被组合到用于训练模型的训练数据集中](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e855d8983569b56c80d1d_image7.png)


要创建托管的训练数据集，用户需要为Pandas，Numpy或Spark数据框提供特征，标签和元数据。

###### 创建一个托管的训练数据集
```py
from hops import featurestore

features_df = featurestore.get_features(["average_attendance", "average_player_age"])

featurestore.create_training_dataset(
   features_df,
   training_dataset_name,
   training_dataset_description,
   computation_job,
   training_dataset_version,
   data_format="tfrecords"
)
```
创建训练数据集后，就可以在特征注册表中发现该数据集，并且用户可以使用它来训练模型。下面是一个示例代码片段，用于使用以HopfFS上的tfrecords格式以分布式方式存储的训练数据集来训练模型。

###### 使用训练数据集训练模型
```py
from hops import featurestore

import tensorflow as tf

dataset_dir = featurestore.get_training_dataset_path(td_name)

# the tf records are written in a distributed manner using partitions

input_files = tf.gfile.Glob(dataset_dir + "/part-r-*")

# tf record schemas are managed by the feature store
‍
tf_record_schema = featurestore.get_training_dataset_tf_record_schema(td_name)

# tf records are a sequence of *binary* (serialized with protobuf) records that need to be decoded.

def decode(example_proto):
‍
   return tf.parse_single_example(example_proto, tf_record_schema)

dataset = tf.data.TFRecordDataset(input_files)
   .map(decode)
   .shuffle(shuffle_buffer_size)
   .batch(batch_size)
   .repeat(num_epochs)

# three layer MLP for regression

model = tf.keras.Sequential([
   layers.Dense(64, activation="relu"),
   layers.Dense(64, activation="relu"),
   layers.Dense(1)
])

model.compile(optimizer=tf.train.AdamOptimizer(lr), loss="mse")

model.fit(dataset, epochs=num_epochs, steps_per_epoch=spe)
```


## 特征注册表

特征注册表是用于发布和发现特征以及训练数据集的用户界面。特征注册表还用作通过比较特征版本来分析特征随时间演变的工具。启动新的数据科学项目时，项目中的数据科学家通常从扫描特征注册表中的可用特征开始，并仅为其模型添加FeatureStore中尚不存在的新特征。

![图12\. Hopsworks上的Feature Registry](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e8692240e833f04a8662c_image10.png)


特征注册表提供：

*   在特征/特征组/训练数据集元数据上的关键字搜索。
*   在特征/特征组/训练数据集元数据上创建/更新/删除/查看操作。
*   自动特征分析。
*   特征依赖关系跟踪。
*   特征作业跟踪。
*   特征数据预览。

### 自动特征分析

在要素存储中更新要素组或训练数据集时，将执行数据分析步骤。特别是，我们着眼于聚类分析，特征相关性，特征直方图和描述性统计。我们发现这些是最常见的统计类型，我们的用户在特征建模阶段会发现它们有用。例如，特征相关信息可用于识别冗余特征，特征直方图可用于监视特征的不同版本之间的特征分布，以发现协变量偏移，聚类分析可用于发现异常值。在特征注册表中访问此类统计信息有助于用户决定要使用哪些特征。

![图13.使用特征注册表查看训练数据集的特征相关性](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e86b88f10c5ca89ff63dc_image4.png)



‍

![图14.使用特征注册表查看特征组中特征的分布](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e86b8240e83d611a86684_image15.png)



### 特征依赖树和自动回滚

当FeatureStore区的大小增加时，应自动安排作业以重新计算特征，以避免潜在的管理瓶颈。Hopsworks要素存储中的要素组和训练数据集已链接到Spark / Numpy / Pandas作业，从而可以在必要时复制和重新计算要素。此外，每个特征组和训练数据集可以具有一组数据依赖性。通过将要素组和训练数据集链接到作业和数据依存关系，可以使用工作流管理系统（如[Airflow [10]）](https://www.google.com/url?q=https://airflow.apache.org/&sa=D&ust=1546167503239000)自动回填Hopsworks要素存储中的要素。

![图15.特征依赖关系跟踪](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e86e78f10c5134eff690e_image11.png)



## FeatureStore服务的多用户权限管理

我们认为，特征库的最大好处在于将其集中在整个组织中。要素存储中可用的高质量要素越多，越好。例如，Uber在2017年报告说，他们[的特征库中大约有10000个特征[11]](https://www.google.com/url?q=https://eng.uber.com/michelangelo/&sa=D&ust=1546167503240000)。

尽管有集中特征的好处，但我们发现有必要对特征执行访问控制。我们与之交谈的几个组织正在部分地处理敏感数据，这些数据需要特定的访问权限，而该访问权限并未授予组织中的每个人。例如，将从敏感数据中提取的特征发布到组织内公开的FeatureStore可能不可行。

为了解决这个问题，我们利用内置在[Hopsworks平台](https://www.google.com/url?q=https://www.logicalclocks.com/introducing-hopsworks/&sa=D&ust=1546167503240000)架构中的多租户属性[12]。默认情况下，Hopsworks中的特征部件存储是项目专用的，并且可以在项目之间共享，这意味着组织可以合并公共特征部件和专用特征部件存储。组织可以具有与组织中的每个人共享的中央公共FeatureStore，以及包含敏感性质的特征的私有FeatureStore，只有具有适当权限的用户才能访问该特征。

![图16.根据组织需求，可以将要素分为几个要素存储，以保留数据访问控制](https://uploads-ssl.webflow.com/5cd1425c57508f884f7c7e4a/5d5e870b240e83aae3a8684f_image12.png)


## 未来的工作

本博客文章中介绍的FeatureStore是所谓的_批处理_FeatureStore，这意味着它是为训练和非实时模型服务而设计的FeatureStore。在以后的工作中，我们计划扩展FeatureStore，以满足在提供面向用户的模型期间所需的实时保证。此外，我们目前正在评估特征设计对领域特定语言（DSL）的需求。通过使用DSL，不精通Spark / Pandas / Numpy的用户可以提供抽象的声明性描述，说明应如何从原始数据中提取特征，然后该库将该描述转换为Spark作业以计算特征。最后，我们也正在研究支持[Petastorm [13]](https://www.google.com/url?q=https://github.com/uber/petastorm&sa=D&ust=1546167503241000)作为训练数据集的数据格式。通过将训练数据集存储在Petastorm中，我们可以有效地将Parquet数据直接输入到机器学习模型中。我们认为Petastorm是tfrecord的潜在替代品，与Tensorflow（例如PyTorch）相比，它可以使针对其他ML框架的训练数据集重用更为容易。




