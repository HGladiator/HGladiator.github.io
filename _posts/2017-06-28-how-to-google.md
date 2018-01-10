---
layout: post
title: 活用搜索工具
categories: 工具
tags: Python
excerpt: 你还在用百度吗？你还在用中文搜索吗？面向Google编程时搜索是不是花了你很久时间？
---


* content
{:toc}
 
国内搜索环境差强人意，一不小心就能要人命，    
大多数结果从来不是解决问题，而是扩大你的问题，最后一举解决掉你的钱包。   
在面向Google编程时也会经常遇到很多问题找不到解决答案，   
或者是答非所问，这时就需要学会活用搜索引擎了。   

## Python交互模式下  多行代码复制输入

因为在复制的时候会变成不连续的多行，经常会出现各种代码断裂，可是代码多了又不能一行一行敲进去，太浪费时间了。
这是一个很简单的问题，当遇到问题时，

1. 先用中文搜索一下
![](/img/2017/中文搜索.png)
可以看到有很多结果，但是点进去会发现没有解答问题，这个时候会想去专业的社区搜索，比如Stack Overflow，对于我平时只用Google、Stack Overflow和博客园，工具不在多，在遇到问题时能够利用工具快速有效解决问题就够了。

2. 我用英文搜同样的问题
![](/img/2017/english_search.png)

3. 第一个结果就解决了问题
![](/img/2017/answer.png)

4. 使用多行字符输入，最后再把字符串开头去掉就可以了，是不是非常的简单。

5. 总结
这时候就会发现，国内的内容同质化严重，各种抄袭多如牛毛。
上次看到一个文章，一眼就看出是抄的，虽然没有著名转载，写的是原创，因为文章质量很高，我就试着找找文章原来的主人，最后还是被我找到了原作者，发现原作者是360里力推OpenResty的那个大佬。

国内搜索解决问题的效率略低，这也就是我使用Google、Stack Overflow的原因，这个就是一个典型的例子，所以老铁们要是用中文搜索解决不了问题的时候，
可以试试用英文搜索。


## 其他的技巧
因为搜索引擎现在还没有发展到非常的人性化，会有一些其他的辅助可以让搜索引擎更好的理解你的问题。

1. 【双引号】  双引号包含的字段为完全匹配搜索，搜索结果中必定都包含不拆分的搜索词句　　　
2. 【减号】  减号后的关键字是搜索结果中不包含减号后的关键字　　
3. 【星号】 星号是指通配符，代表任何字词（一个星号就可以代表好多字符），比如马*梅　　
4. 【 site:】 ```eg. site:v2ex.com``` 只搜索指定某网站内里的关键词，可以搜索一些会员才能搜索站内信息的网站
5. 【 inurl:】 按 链接搜索返回那些网址url里面包含你指定关键词的页面。例如“inurl: passwd” ，返回那些网址url里面包含你指定关键词passwd的页面
6. 【 intitle: 】按标题搜索帮助google限定了你搜索的结果，只有那些标题含有你指定的关键词的页面会返回给你。

这里是常用的，还有很多其他的技巧，现在就去用以上技巧搜搜看。