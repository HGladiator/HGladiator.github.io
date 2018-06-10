---
layout: post
title: 搭建私人GitLab服务器
categories: 搭建环境
tags: Git
excerpt: 薅羊毛拿下的服务器放着吃灰，还不如用来搭建一个私人的GitLab服务器玩玩
---

* content
{:toc}


## GitLab架构

1、前端：Nginx,用于git tool走http或https协议  
2、后端：gitlab服务，采用ruby on Rails框架，通过unicorn实现后台服务及多进程   
3、SSHD：开启sshd服务，用于用户上传ssh key进行版本克隆及上传。用户上传的ssh key是保存到git账户中  
4、数据库：目前仅支持MySQL和PostgreSQL  
5、Redis：用于存储用户session和任务，任务包括新建仓库、发送邮件等等  
6、Sidekiq：Rails框架自带的，订阅redis中的任务并执行  


##  GitLab资料

官网部署说明：
1. [官方中文说明](https://www.gitlab.com.cn/installation/)
2. [官方手册](http://doc.gitlab.com/omnibus/)


官网支持的方式：
1. [包含一切的rpm包（官方推荐,也是本文安装方式）](https://about.gitlab.com/downloads/)
2. [手动安装（深入了解](https://gitlab.com/gitlab-org/gitlab-ce/blob/master/doc/install/installation.md)
3. [第三方docker镜像](https://github.com/sameersbn/docker-gitlab)

## 安装配置依赖项

如果已经安装了Postfix来发送邮件请在安装期间选择 `Internet Site`. 你也可以用Sendmail或者使用自定义的SMTP服务器来代替Postfix. 如果希望使用 Exim, 请把它当做SMTP来配置.在Centos 6和7上, 下面的命令也会配置系统防火墙,把HTTP和SSH端口开放.

```
sudo yum install curl policycoreutils openssh-server openssh-clients  postfix cronie
sudo systemctl enable sshd
sudo systemctl start sshd
sudo systemctl enable postfix
sudo systemctl start postfix
sudo firewall-cmd --permanent --add-service=http
sudo systemctl reload firewalld
```

## 添加并安装GitLab软件包

```
curl http://packages.gitlab.cc/install/gitlab-ce/script.rpm.sh |  sudo bash
yum install gitlab-ce
```

如果不习惯这种通过管道命令安装的方式,可以在这里找到完整的安装脚本.或者你可以选择对应系统的GitLab安装包 并使用下面的命令进行安装
```
curl -LJO http://mirror.tuna.tsinghua.edu.cn/gitlab-ce/yum/el6/gitlab-ce-XXX.rpm/download
rpm -i gitlab-ce-XXX.rpm
```

（推荐使用gitlab-ce镜像站，下载后保留使用300M左右）
[清华yum源](https://mirror.tuna.tsinghua.edu.cn/gitlab-ce/)

## 配置和使用GitLab
```
gitlab-ctl reconfigure
```


## 修改域名

默认的配置文件保存在 /etc/gitlab/gitlab.rb ，执行：

```
# 打开文件
vim /etc/gitlab/gitlab.rb
# 修改
external_url 'http://your.domain'
# 执行
gitlab-ctl reconfigure
```

有一定概率出现 502 错误，刷新浏览器或者再次更新配置即可。

## 界面汉化（测试通过）

问题：保证版本一致性，否则上传代码会触发bug
由于服务对象是广大师生，为了降低新手上手的难度，所有进行汉化也是非常有必要的。好在国内有人已经进行了这方面的工作，我们只需要共享其成果即可（欢迎向原项目提交高质量翻译）。
首先确认版本：
```
cat /opt/gitlab/embedded/service/gitlab-rails/VERSION
```

并确认当前汉化版本的VERSION 是否相同，当前最新的汉化版本为8.6。如果安装版本小于当前汉化版本，请先升级。如果安装版本大于当前汉化版本，请在本项目中提交新的issue。如果版本相同，首先在本地 clone 仓库。
```
# GitLab.com 仓库
git clone https://gitlab.com/larryli/gitlab.git

# Coding.net 镜像
git clone https://git.coding.net/larryli/gitlab.git
```
根据网友的测试，Coding.net的镜像不完整，clone之后无法checkout。然后比较汉化分支和原分支，导出 patch用的diff文件。

```
# 8.1 版本的汉化补丁
git diff origin/8-6-stable..8-6-zh > ../8.6.diff   （已经生成可以直接进行下面的操作）
```

然后上传 8.6.diff 文件到服务器。

```
#　停止 gitlab
gitlab-ctl stop
patch -d /opt/gitlab/embedded/service/gitlab-rails -p1 < 8.6.diff
```

确定没有.rej文件，重启 GitLab即可。
```
gitlab-ctl start
```
如果汉化中出现问题，请重新安装 GitLab（ 注意备份数据 ）。


## GitLab管理

### 启动、停止、重启组件

```
＃　启动所有 GitLab 组件
gitlab-ctl start    
＃　停止所有 GitLab 组件
gitlab-ctl stop   
＃　重启所有 GitLab 组件
gitlab-ctl restart  
```

### 常用管理命令

#### 相关操作
```
gitlab-ctl reconfigure       如果更改了主配置文件 [gitlab.rb文件],使配置文件生效 但是会初始化除了gitlab.rb之外的所有文件
gitlab-ctl show-config       验证配置文件
gitlab-ctl restart           重启gitlab服务
gitlab-ctl stop unicorn      停止组件内某一个服务
gitlab-ctl status unicorn    查看状态
gitlab-ctl kill unicorn      kill掉某一个服务
gitlab-ctl status unicorn    再次查看状态
gitlab-ctl start unicorn     启动服务
gitlab-rake gitlab:env:info  系统信息监测
gitlab-rake gitlab:check     各种状态监测
```


#### 相关目录
```
/var/opt/gitlab/git-data/repositories/root 库默认存储目录
/opt/gitlab                                是gitlab的应用代码和相应的依赖程序
/var/opt/gitlab                            此目录下是运行gitlab-ctl reconfigure命令编译后的应用数据和配置文件，不需要人为修改配置
/etc/gitlab                                此目录下存放了以omnibus-gitlab包安装方式时的配置文件，这里的配置文件才需要管理员手动编译配置
/var/log/gitlab                            此目录下存放了gitlab各个组件产生的日志
/var/opt/gitlab/backups/                   备份文件生成的目录
```


## 备份和恢复

###  备份

备份GitLab repositories and GitLab metadata
在 crontab 中加入如下命令：

```
0 2 * * * /usr/bin/gitlab-rake gitlab:backup:create
```

注：备份文件是一个归档文件，且开头是unix时间
在实际的生产环境中备份策略建议：本地保留三天，在异地备份永久保存
这里以unix时间戳来标记备份的时间，这个时间戳对人来说不好读懂，可使用date命令把其转换成人可读的格式，如下：

```
#date -d @1529637013
2018年 06月 22日 星期五 11:10:13 CST
```


```
＃ 修改备份目录
vim /etc/gitlab/gitlab.rb

gitlab_rails['backup_path'] = "/data/git-backups"
# limit backup lifetime to 7 days - 604800 seconds
gitlab_rails['backup_keep_time'] = 604800

# 创建备份目录，修改属主和属组
mkdir /data/git-backups
chown -R git.git /data/git-backups
```
手动进行一次备份，测试一下备份是否有效，测试备份正常，添加定时任务即可

###  恢复

首先进入备份 GitLab 的目录，这个目录是配置文件中的 `gitlab_rails[‘backup_path’]`，默认为 `/var/opt/gitlab/backups` 。然后停止 unicorn 和 sidekiq ，保证数据库没有新的连接，不会有写数据情况。

```
sudo gitlab-ctl stop unicorn
# ok: down: unicorn: 0s, normally up
sudo gitlab-ctl stop sidekiq
# ok: down: sidekiq: 0s, normally up

# 然后恢复数据，1529637013为备份文件的时间戳
gitlab-rake gitlab:backup:restore BACKUP=1529637013
```

## 邮箱设置

1. GitLab中使用postfix进行邮件发送。因此，可以卸载系统中自带的sendmail。
使用`yum list installed`查看系统中是否存在sendmail，若存在，则使用`yum remove sendmail`指令进行卸载。
2. 测试系统是否可以正常发送邮件。

```
echo "Test mail from postfix" ####  mail -s "Test Postfix" xxx@xxx.com
```


注：上面的`xxx@xxx.com`为你希望收到邮件的邮箱地址。

当邮箱收到系统发送来的邮件时，将系统的地址复制下来，如：`root@iZ23syflhhzZ.localdomain`,打开`/etc/gitlab/gitlab.rb`,将  
`gitlab_rails['gitlab_email_from'] = 'gitlab@example.com' `  
修改为  
`gitlab_rails['gitlab_email_from'] = 'root@iZ23syflhhzZ.localdomain'  `   
保存后，执行`sudo gitlab-ctl reconfigure`重新编译GitLab。如果邮箱的过滤功能较强，请添加系统的发件地址到邮箱的白名单中，防止邮件被过滤。
Note:系统中邮件发送的日志可通过`tail /var/log/maillog`命令进行查看。


# FQA

1. 在浏览器中访问GitLab出现502错误  
原因：内存不足。  
解决办法：检查系统的虚拟内存是否随机启动了，如果系统无虚拟内存，则增加虚拟内存，再重新启动系统。  
2. 80端口冲突  
原因：Nginx默认使用了80端口。 
解决办法：为了使Nginx与Apache能够共存，并且为了简化GitLab的URL地址，Nginx端口保持不变，修改Apache的端口为4040。这样就可以直接用使用ip访问Gitlab。而禅道则可以使用4040端口进行访问，像这样：`xxx.xx.xxx.xx:4040/zentao`。具体修改的地方在`/etc/httpd/conf/httpd.conf`这个文件中，找到Listen 80这一句并将之注释掉，在底下添加一句Listen 4040，保存后执行`service httpd restart`重启apache服务即可。

3. 8080端口冲突
原因：由于unicorn默认使用的是8080端口。
解决办法：打开`/etc/gitlab/gitlab.rb`,打开`# unicorn[‘port’] = 8080`的注释，将8080修改为9090，保存后运行`sudo gitlab-ctl reconfigure`即可。
4. STMP设置
配置无效，暂时不知道原因。
5. GitLab头像无法正常显示
原因：gravatar被墙
解决办法：
编辑`/etc/gitlab/gitlab.rb`，将
`#gitlab_rails['gravatar_plain_url'] = 'http://gravatar.duoshuo.com/avatar/%{hash}?s=%{size}&d=identicon'` 
注释去掉
然后在命令行执行： 
```
gitlab-ctl reconfigure 
gitlab-rake cache:clear RAILS_ENV=production
```

