---
layout: post
title:  基于Docker的Spark集群环境搭建
categories: Docker
tags: Docker
excerpt: 基于Docker的Spark集群环境搭建，环境都是最新的稳定方案，网上大多都过时了，要么就是有各种问题，不过还有有很多值得借鉴的地方，我自己手动搭建了一遍,
---

* content
{:toc}

### 集群环境基本介绍
#### 软件版本
- 系统环境: CentOS 6
- Java: OpenJDK 8
- Hadoop: 2.7.4
- Spark: 2.2.1
- Hive: 2.3.2


#### 镜像依赖关系
- 基于docker-compose管理镜像和容器，并进行集群的编排 
- 软件都是通过阿里云镜像下载的，有一个Hadoop的native包用的是在Github上找到的资源。 

### 集群操作
#### 获取镜像
可以下载Dockerfile 自己动手制作，[这是项目](https://github.com/HGladiator/Docker-Spark)，这里有很多我在网上找到的做法，我自己做了一些改进，也可以下载我制作好了的镜像(网不好，上传很多次都超时。。。)   
#### 集群操作
下面所有的操作都是在放docker-compose.xml的目录下输入，其他路径都是无法使用集群的    

1. 初始化工作
```
#创建容器
docker-compose up -d
#格式化HDFS。第一次启动集群前，需要先格式化HDFS；以后每次启动集群时，都不需要再次格式化HDFS
docker-compose exec spark-master hdfs namenode -format
#初始化Hive数据库。仅在第一次启动集群前执行一次
docker-compose exec spark-master schematool -dbType mysql -initSchema
```
2. 启动集群，依次执行：
```
#启动容器集群
docker-compose start
#启动HDFS
docker-compose exec spark-master $HADOOP_HOME/sbin/start-dfs.sh
#启动YARN
docker-compose exec spark-master $HADOOP_HOME/sbin/start-yarn.sh
#启动Spark
docker-compose exec spark-master $SPARK_HOME/sbin/start-all.sh
```
3. 停止集群，依次执行：
```
#停止Spark
docker-compose exec spark-master $SPARK_HOME/sbin/stop-all.sh
#停止YARN
docker-compose exec spark-master $HADOOP_HOME/sbin/stop-yarn.sh
#停止HDFS
docker-compose exec spark-master $HADOOP_HOME/sbin/stop-dfs.sh
#停止容器集群
docker-compose stop
#删除容器集群
docker-compose down
```
---
### 遇到的问题

* 用centos 7会遇到很多问题，查阅资料[解决起来比较麻烦](http://dockone.io/question/729)，后续还会有别的问题，能通过更换版本避免的问题，我都是直接更换版本，这个问题是start sshd error -> [Failed to get D-Bus connection: Operation not permitted](https://serverfault.com/questions/824975/failed-to-get-d-bus-connection-operation-not-permitted)，其实现在都是在基础镜像上启用systemd支持的镜像，没有尝试，最近被各种问题烦到懵逼了。。

* 免密登录的时候发现 没有开启22监听，导致集群一直无法启动,又重新制作了一遍

* Dockerfile 中ADD的命令疑问， 做镜像都不是一次就做成功了的，每次都要下载会比较浪费时间，ADD虽然有自动解压的功能，但是内部实现似乎并不会删除解压前的文件，所以生成的镜像会很大（实测过大小），而COPY命令是无法在解压后删除的，所以用下载的方式制造镜像是最小的，用curl命令直接可以达到下载并解压的效果，没有中间的步骤，下载之后也没有删除文件这样多余的操作，

* hive 的问题，我这里是用远程模式搭建的Hive
FAILED: SemanticException org.apache.hadoop.hive.ql.metadata.HiveException: java.lang.RuntimeException: Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient