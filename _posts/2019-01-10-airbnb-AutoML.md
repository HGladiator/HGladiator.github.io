---
layout: post
title: 使用机器学习预测Airbnb上房屋的价值
categories: AI中台
tags: AutoML
excerpt: 最近，Airbnb机器学习基础设施的进步大大降低了将新的机器学习模型部署到生产环境的成本。例如，我们的ML Infra团队建立了一个通用的功能库，该功能库使用户可以在模型中利用高质量，经过审查的可重用功能。数据科学家已开始将几种AutoML工具整合到他们的工作流程中，以加快模型选择和性能基准测试的速度。此外，ML infra创建了一个新框架，该框架将把Jupyter笔记本自动转换为Airflow pipelines。
mathjax: false
date: 2019-01-10
---



* content
{:toc}

# 介绍

数据产品一直是Airbnb服务的重要组成部分。但是，我们早就认识到制造数据产品的成本很高。例如，个性化搜索排名使宾客能够更轻松地发现房屋，而智能定价则使房东可以根据供需设置更具竞争力的价格。但是，这些项目每个都需要大量的专用数据科学和工程时间和精力。

最近，Airbnb机器学习基础设施的进步大大降低了将新的机器学习模型部署到生产环境的成本。例如，我们的ML Infra团队建立了一个通用的功能库，该功能库使用户可以在模型中利用高质量，经过审查的可重用功能。数据科学家已开始将几种AutoML工具整合到他们的工作流程中，以加快模型选择和性能基准测试的速度。此外，ML infra创建了一个新框架，该框架将把Jupyter笔记本自动转换为Airflow pipelines。

在这篇文章中，我将描述这些工具如何协同工作以加快建模过程，从而降低LTV建模的特定用例的总体开发成本-预测Airbnb上房屋的价值。

# 什么是LTV？

客户生命周期价值（LTV）是电子商务和市场公司中流行的概念，它捕获了固定时间范围内用户的预期价值，通常以美元为单位。

在Spotify或Netflix等电子商务公司中，LTV通常用于制定定价决定，例如设置订阅费用。在Airbnb这样的市场公司中，了解用户的LTV可以使我们更有效地在不同的营销渠道上分配预算，基于关键字为在线营销计算更精确的出价，并创建更好的列表细分。

尽管人们可以使用过去的数据来[计算](https://medium.com/swlh/diligence-at-social-capital-part-3-cohorts-and-revenue-ltv-ab65a07464e1)现有列表[的历史价值](https://medium.com/swlh/diligence-at-social-capital-part-3-cohorts-and-revenue-ltv-ab65a07464e1)，但我们又进一步采用机器学习来预测新列表的LTV。

# 用于LTV建模的机器学习工作流程

数据科学家通常习惯于与机器学习相关的任务，例如功能工程，原型设计和模型选择。但是，将模型原型投入生产通常需要一组正交的数据工程技能，而数据科学家可能不熟悉这些技能。

![](https://miro.medium.com/max/2273/1*zT1gNPErRqizxlngxXCtBA.png)

幸运的是，在Airbnb，我们拥有机器学习工具，可以将产生ML模型的工程工作抽象化。实际上，如果没有这些出色的工具，我们就无法将模型投入生产。这篇文章的其余部分分为四个主题，以及我们用来解决每个任务的工具：

*   **特征工程：**定义相关特征
*   **原型设计和培训：**训练模型原型
*   **模型选择和验证：**执行模型选择和调整
*   **量产：**将选定的模型原型投入生产

# 特征工程

> **使用的工具：Airbnb的内部功能库-Zipline**

任何受监督的机器学习项目的第一步之一就是定义与所选结果变量相关的相关特征，此过程称为特征工程。例如，在预测LTV时，可以计算下一个可用日历日的180个日历日期或相对于同一市场中可比较列表的价格。

在Airbnb，功能工程通常意味着编写Hive查询以从头开始创建功能。但是，这项工作很繁琐且耗时，因为它需要特定的领域知识和业务逻辑，这意味着功能管线通常不易于共享甚至不可重用。为了使这项工作更具可扩展性，我们开发了**Zipline**，这是一个培训功能库，提供了不同粒度级别的功能，例如在主机，来宾，列表或市场级别。

这种内部工具的**众包**性质使数据科学家可以使用其他人为过去的项目准备的各种高质量，经过审查的功能。如果所需的功能不可用，则用户可以使用功能配置文件来创建自己的功能，如下所示：
```yaml
source: {
  type: hive
  query:"""
    SELECT
      id_listing as listing
      , dim_city as city
      , dim_country as country
      , dim_is_active as is_active
      , CONCAT(ds, ' 23:59:59.999') as ts
    FROM
      core_data.dim_listings
    WHERE
      ds BETWEEN '{{ start_date }}' AND '{{ end_date }}'

  """
  dependencies: [core_data.dim_listings]
  is_snapshot: true
  start_date: 2010-01-01
}

features: {
  city: "City in which the listing is located."
  country: "Country in which the listing is located."
  is_active: "If the listing is active as of the date partition."
}
```

当构建训练集需要多个功能时，Zipline将自动执行智能键联接并在后台回填训练数据集。对于上市的LTV模型，我们使用了现有的Zipline功能，还添加了一些我们自己的功能。总之，我们的模型中有150多个功能，其中包括：

*   **位置**：国家，市场，邻里和各种地理特征
*   **价格**：每晚价格，清洁费，相对于类似列表的价格点
*   **可用性**：总可用夜数，手动阻止的夜数百分比
*   **可预订性**：过去X天内的预订或夜数
*   **质量**：评论评分，评论数量和便利设施
		

![训练数据集示例](https://miro.medium.com/max/1679/1*KYs7WNNfdwKmKcVbgKGkiw.png)


通过定义特征和结果变量，我们现在可以训练模型以从历史数据中学习。

# 原型设计与培训

> **使用的工具：Python中的机器学习库— **[**scikit-learn**](http://scikit-learn.org/stable/)

与上面的示例训练数据集一样，在适合模型之前，我们通常需要执行其他数据处理：

*   **数据估算：**我们需要检查是否有任何数据丢失，以及是否随机丢失了该数据。如果没有，我们需要调查原因并了解根本原因。如果是，我们应该估算缺失的值。
*   **编码分类变量**：通常，我们无法在模型中使用原始类别，因为模型不知道如何适合字符串。当类别数较少时，我们可以考虑使用[单热编码](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)。但是，当基数较高时，我们可以考虑使用[序数编码](https://www.kaggle.com/general/16927)，即按每个类别的频率计数进行编码。

在这一步中，我们不太清楚要使用的最佳功能是什么，因此编写允许我们快速迭代的代码至关重要。管道构造通常在[Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)和[Spark](https://spark.apache.org/docs/latest/ml-pipeline.html)等开源工具中可用，是一种非常方便的原型制作工具。管道使数据科学家可以指定高级蓝图，这些蓝图描述了应如何转换要素以及要训练哪些模型。为了更具体，下面是我们的LTV模型管道中的代码片段：

```py
transforms = []

transforms.append(
    ('select_binary', ColumnSelector(features=binary))
)

transforms.append(
    ('numeric', ExtendedPipeline([
        ('select', ColumnSelector(features=numeric)),
        ('impute', Imputer(missing_values='NaN', strategy='mean', axis=0)),
    ]))
)

for field in categorical:
    transforms.append(
        (field, ExtendedPipeline([
            ('select', ColumnSelector(features=[field])),
            ('encode', OrdinalEncoder(min_support=10))
            ])
        )
    )
    
features = FeatureUnion(transforms)
```

在较高的层次上，我们使用管道为不同类型的要素指定数据转换，具体取决于这些要素是二进制，分类还是数字类型。[最后的FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)只需按列组合功能即可创建最终的训练数据集。

用管道编写原型的好处是，它可以使用[数据转换](http://scikit-learn.org/stable/data_transforms.html)来提取乏味的[数据转换](http://scikit-learn.org/stable/data_transforms.html)。总而言之，这些转换可确保在训练和评分之间可以一致地转换数据，从而解决了将原型转换为生产时数据转换不一致的普遍问题。

此外，管道还将数据转换与模型拟合分开。尽管上面的代码中未显示，但数据科学家可以添加最后一步来指定用于模型拟合的[估计量](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)。通过探索不同的估计量，数据科学家可以执行模型选择以选择最佳模型，以改善模型的样本外误差。

# 执行模型选择

> **使用的工具：各种**[**AutoML**](https://medium.com/airbnb-engineering/automated-machine-learning-a-paradigm-shift-that-accelerates-data-scientist-productivity-airbnb-f1f8a10d61f8)**框架**

如前一节所述，我们需要确定哪种候选模型最适合投入生产。要做出这样的决定，我们需要权衡模型可解释性和模型复杂性之间的权衡。例如，稀疏线性模型可能很容易解释，但不够复杂，无法很好地概括。基于树的模型可能足够灵活，可以捕获非线性模式，但无法很好地解释。这就是所谓的[**Bias-Variance权衡**](http://scott.fortmann-roe.com/docs/BiasVariance.html)。


![](https://miro.medium.com/max/1316/1*tQbBEq6T8ZJ9lFSCbZKFqw.png)

James，Witten，Hastie和Tibshirani从R的《统计学习入门》中引用的图

在诸如保险或信用审查之类的应用程序中，模型必须是可解释的，因为对于模型而言，避免无意中歧视某些客户非常重要。但是，在诸如图像分类的应用程序中，具有性能分类器比可解释模型重要得多。

考虑到模型选择可能非常耗时，我们尝试使用各种[AutoML](https://medium.com/airbnb-engineering/automated-machine-learning-a-paradigm-shift-that-accelerates-data-scientist-productivity-airbnb-f1f8a10d61f8)工具来加快过程。通过探索各种各样的模型，我们发现哪种类型的模型往往表现最佳。例如，我们了解到，[eXtreme梯度增强树](https://github.com/dmlc/xgboost)（XGBoost）明显优于基准模型，例如均值响应模型，岭回归模型和单决策树。


![](https://miro.medium.com/max/1530/1*y1O7nIxCFmgQamCfsrWfjA.png)

比较RMSE可让我们执行模型选择

鉴于我们的主要目标是预测上市价值，因此使用XGBoost生产最终模型感到很自在，因为它倾向于灵活性而不是可解释性。

# 将模型原型投入生产

> **使用的工具：Airbnb的笔记本翻译框架-ML Automator**

正如我们之前提到的，建立生产管道与在本地笔记本电脑上构建原型有很大不同。例如，我们如何执行定期再训练？我们如何有效地评分大量示例？我们如何建立流水线来监控模型的性能？

在Airbnb，我们建立了一个名为**ML Automator**的框架，该框架可自动将Jupyter笔记本电脑转换为[Airflow](https://medium.com/airbnb-engineering/airflow-a-workflow-management-platform-46318b977fd8)机器学习管道。该框架是专门为数据科学家设计的，这些科学家已经熟悉使用Python编写原型，并希望将其模型应用于数据工程方面的经验有限。


![](https://miro.medium.com/max/1787/1*uLCH5Ozfj8mM07bKXIg20Q.png)

ML Automator框架的简化概述（图片来源：Aaron Keys）

*   首先，该框架要求用户在笔记本中指定模型配置。该模型配置的目的是告诉框架在哪里可以找到训练表，为训练分配多少计算资源以及如何计算分数。
*   另外，数据科学家需要编写特定的_拟合_和_变换_函数。fit函数指定如何精确地进行训练，并且transform函数将包装为Python UDF以进行分布式评分（如果需要）。

这是一个代码片段，展示了如何在LTV模型中定义_拟合_和_变换_函数。fit函数告诉框架将训练XGBoost模型，并且将根据我们先前定义的管道执行数据转换。

```py
def fit(X_train, y_train):
    import multiprocessing
    from ml_helpers.sklearn_extensions import DenseMatrixConverter
    from ml_helpers.data import split_records
    from xgboost import XGBRegressor

    global model
    
    model = {}
    n_subset = N_EXAMPLES
    X_subset = {k: v[:n_subset] for k, v in X_train.iteritems()}
    model['transformations'] = ExtendedPipeline([
                ('features', features),
                ('densify', DenseMatrixConverter()),
            ]).fit(X_subset)
    
    # apply transforms in parallel
    Xt = model['transformations'].transform_parallel(X_train)
    
    # fit the model in parallel
    model['regressor'] = XGBRegressor().fit(Xt, y_train)
        
def transform(X):
    # return dictionary
    global model
    Xt = model['transformations'].transform(X)
    return {'score': model['regressor'].predict(Xt)}        
```

笔记本合并后，ML Automator会将经过训练的模型包装在[Python UDF中，](http://www.florianwilhelm.info/2016/10/python_udf_in_hive/)并创建如下所示的[Airflow](https://airflow.incubator.apache.org/)管道。数据工程任务（例如数据序列化，定期重新培训的计划和分布式评分）都封装为该日常批处理工作的一部分。结果，此框架大大降低了数据科学家的模型开发成本，就好像有专门的数据工程师与数据科学家一起将模型投入生产！


![](https://miro.medium.com/max/1670/1*DvPE_V_SoHV3pikOqiZxsg.png)

正在生产中运行的LTV Airflow DAG的图形视图

**注意：** _除了生产化之外，还有其他主题，例如随着时间的推移跟踪模型性能或利用弹性计算环境进行建模，我们将不在本文中介绍。放心，这些都是正在开发的活跃领域。_

# 经验教训与展望

在过去的几个月中，数据科学家与ML Infra进行了非常紧密的合作，这种合作产生了许多很棒的模式和想法。实际上，我们相信这些工具将为Airbnb上如何开发机器学习模型开辟新的范式。

*   **首先，模型开发的成本大大降低**：通过结合各个工具的不同优势：用于功能工程的Zipline，用于模型原型的管道，用于模型选择和基准测试的AutoML，最后是用于生产的ML Automator，我们大大缩短了开发周期。
*   **其次，笔记本驱动的设计减少了进入的障碍**：对框架不熟悉的数据科学家可以立即访问大量现实生活中的示例。保证生产中使用的笔记本电脑是正确的，可自动记录文档且是最新的。这种设计推动了新用户的广泛采用。
*   **结果，团队更愿意在ML产品创意上进行投资**：在撰写本文时，我们还有其他几个团队通过类似的方法来探索ML产品创意：确定清单检查队列的优先级，预测清单将进行的可能性添加共同主持人，并自动标记低质量的列表。

我们对该框架的未来及其带来的新范例感到非常兴奋。通过弥合原型与生产之间的差距，我们可以真正地使数据科学家和工程师从事端到端机器学习项目，并使我们的产品更好。

* * *

_是否要使用或构建这些ML工具？也可以加入Airbnb[_人才加入我们的数据科学和分析团队_](https://www.airbnb.com/careers/departments/data-science-analytics)_！_