---
layout: post
title:  Python3连接Hbase
categories: Python
tags: HBase
excerpt: Python3连接Hbase的方式，而网上搜到的都是些旧版本的文章，或者已经不适用了
---

* content
{:toc}

搜索网上的Python3连接Hbase的方法,会搜到以下两种方式:
* 通过thrift取连接Hbase  
* 通过Happybase连接Hbase 


### Thrift
thrift连接的时候需要导入一个Hbase包,
实际是需要另外下载一个第三方包hbase-thrift,
这个包是用Python2写的,加载时会出现兼容性问题,比如:   
<div style="text-align:center" markdown="1">
![](/img/2017/SyntaxError.jpg)
![](/img/2017/xrange.jpg)
![Alt Text](/img/2017/TApplicationException.jpg)
</div>

### Happybase
happybase是对thrift的一种封装,[支持Python3](https://github.com/wbolster/happybase/issues/40),
但是在windows下有bug.Linux下没有问题    
<div style="text-align:center" markdown="1">
![thriftparserError](/img/2017/thriftparserError.jpg)
</div>

### Thrift 介绍
> Thrift是一种接口描述语言和二进制通讯协议，它被用来定义和创建跨语言的服务。它被当作一个远程过程调用（RPC）框架来使用，是由Facebook为“大规模跨语言服务开发”而开发的。它通过一个代码生成引擎联合了一个软件栈，来创建不同程度的、无缝的跨平台高效服务，可以使用C#、C++（基于POSIX兼容系统）、Cappuccino、Cocoa、Delphi、Erlang、Go、Haskell、Java、Node.js、OCaml、Perl、PHP、Python、Ruby和Smalltalk。虽然它以前是由Facebook开发的，但它现在是Apache软件基金会的开源项目了。该实现被描述在2007年4月的一篇由Faceb ook发表的技术论文中，该论文现由Apache掌管。      
                                                                                             ----- 来自wikipedia

[thrift对Python3的支持](https://github.com/apache/thrift/pull/213)