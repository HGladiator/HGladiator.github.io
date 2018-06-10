---
layout: post
title:  GitLab的简单使用
categories: 工具
tags: Git
excerpt: GitLab是一个利用Ruby on Rails开发的开源应用程序，实现一个自托管的Git项目仓库，可通过Web界面进行访问公开的或者私人项目。

---

* content
{:toc}


<div style="text-align:center" markdown="1">
![](/img/2018/gitlab_logo.png)
</div>

### 特点
1. 团队版本的GitHub，拥有与Github类似的功能，能够浏览源代码，管理缺陷和注释。
2. 团队权限管理，团队交流沟通。
3. 免费开源，完全可以自己搭建一个私人团队的代码仓库，也可以对其进行封装。开源中国代码托管平台`git.oschina.net`就是基于GitLab项目搭建。
4. 丰富的功能包括Git仓库管理、代码审查、问题跟踪、动态订阅、wiki等功能.以及GitLab内部集成的GitLab CI 更是一个持续集成和交付的好工具。



###  添加新项目

1、选择创建新项目

登录成功后，点击导航条上的 “+” 就可以进入创建项目的页面

![new_project_1.png](/img/2018/gitlab_new_project.png)

2、填写项目的信息

在创建工程的页面，按照要求填写项目的名称和可见性等信息。

（1）Project path：项目的路径，一般不用填，默认会有相应的网址、用户名/组名，

（2）Project name: 项目的名称

（3）Description（项目的描述）：可选项，对项目的简单描述

（4）Visibility Level（项目可见级别）：提供Private（私有的，只有你自己或者组内的成员能访问）/Internal（所有登录的用户）/Public(公开的，所有人都可以访问)三种选项。

### 添加和配置SSH公钥

SSH（Secure Shell）是一种安全协议，在你的电脑与GitLab服务器进行通信时，我们使用SSH密钥（SSH Keys）认证的方式来保证通信安全。你可以在网络上搜索到关于SSH密钥的更多介绍；下面我们重点讲解如何创建 SSH密钥，并将密钥中的公钥添加到GitLab，以便我们通过SSH协议来访问Git仓库。

SSH 密钥的创建需要在终端（命令行）环境下进行，我们首先进入命令行环境。通常在OS X和Linux平台下我们使用终端工具（Terminal），在Windows平台中，可以使用Git Bash工具。

进入命令行环境后，我们执行以下操作来创建 SSH 密钥。

#### 1.进入SSH目录

`cd ~/.ssh`

（1）如果还没有 ~/.ssh 目录，可以手工创建一个(`mkdir ~/.ssh`)，之后再通过`cd ~/.ssh`进入SSH目录

（2）可以通过`ls -l`命令查看SSH目录下的文件，来确认你是否已经生成过SSH密钥；如果SSH目录为空，我们开始第二步，生成 SSH 密钥；如果存在id_rsa.pub这个文件，说明你之前生成过SSH密钥，后面有介绍如何添加多个sshkey

#### 2.生成SSH密钥

我们通过下面的命令生成密钥，请将命令中的`YOUR_EMAIL@YOUREMAIL.COM`替换为你自己的`Email`地址。

`ssh-keygen -t rsa -C "YOUR_EMAIL@YOUREMAIL.COM"`

在SSH生成过程中会出现以下信息，按屏幕的提示操作即可；

```
$ ssh-keygen -t rsa -C "YOUR_EMAIL@YOUREMAIL.COM"
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /Users/USERNAME/.ssh/id_rsa.
Your public key has been saved in /Users/USERNAME/.ssh/id_rsa.pub.
The key fingerprint is:
15:81:d2:7a:c6:6c:0f:ec:b0:b6:d4:18:b8:d1:41:48 YOUR_EMAIL@YOUREMAIL.COM

```

说明：

（1）一般情况下，在命令行中输入密码、口令一类的信息时是没有信息回显的。在我们这一步的操作中，输入passphrase口令时，命令行界面上不会随着键盘敲入密码而有什么反馈。

（2）当提示`Enter passphrase (empty for no passphrase) : `时，可以直接按两次回车键输入一个空的 passphrase；也可以选择输入一个 passphrase 口令，如果此时你输入了一个`passphrase`，请牢记，之后每次提交时都需要输入这个口令来确认。

#### 3.获取SSH公钥信息

SSH密钥生成结束后，你可以在SSH目录下看到私钥`id_rsa`和公钥`id_rsa.pub`这两个文件，不要把私钥文件`id_rsa`的信息透露给任何人。我们可以通过文本编辑器或`cat`命令来查看`id_rsa.pub`公钥信息。

（1）通过编辑器。使用你熟悉的文本编辑器，比如 记事本、Sublime Text等软件打开`id_rsa.pub`，复制里面的所有内容以备下一步使用。

（2）通过cat命令。在命令行中敲入`cat id_rsa.pub`，回车执行后命令行界面中会显示`id_rsa.pub`文件里的内容，复制后在下一步使用。

（3）通过直接使用命令将`id_rsa.pub`文件里的内容复制到剪切板中

*   Windows: `clip < ~/.ssh/id_rsa.pub`

*   Mac: `pbcopy < ~/.ssh/id_rsa.pub`

*   GNU/Linux (requires xclip): `xclip -sel clip < ~/.ssh/id_rsa.pub`

#### 4.添加SSH公钥到gitlab

（1）打开`https://gitlab.com/profile/keys`Profile配置页面，选择SSH Keys.

（2）添加SSH公钥

按照要求填写Title和Key，其中Title是Key的描述信息（如My_work_computer等），Key是上面复制的SSH公钥的内容，直接粘贴到输入框中保存即可，一般title自动更新为key的邮箱。

![](/img/2018/add_sshkey.png)

#### 5.测试SSH连接

`ssh -T git@gitlab.com`

如果连接成功的话，会出现以下信息。

`Welcome to GitLab, USERNAME!`

### 如何同时使用多个SSH公钥

如果你已经有了一套ssh(笔者的电脑上就有好几套如github/bitbucket/gitlab,三者各不一样)，为了保证各个服务能正常使用需要配置多个SSH Key。可以按照以下的步骤来实现多套SSH Key的共同工作：

#### 1.生成SSH密钥

假设你已经有了一套名为id_rsa的公秘钥，将要生成的公秘钥名称为gitlab，你也可以使用任何你喜欢的名字。记得把以下命令中的`YOUR_EMAIL@YOUREMAIL.COM`改为你的`Email`地址

`ssh-keygen -t rsa -C "YOUR_EMAIL@YOUREMAIL.COM" -f ~/.ssh/gitlab`

说明：

（1）`-f`后面的参数是自定义的SSH Key的存放路径，将来生成的公秘钥的名字分别是gitlab.pub和gitlab

（2）其他的和上面生成密钥的步骤相同，只是多了下面的配置的步骤

#### 2.配置自定义的公秘钥名称

在SSH用户配置文件~/.ssh/config中指定对应服务所使用的公秘钥名称，如果没有config文件的话就新建一个(`vim ~/.ssh/config`)，并输入以下内容(可以添加多个)：

```
Host gitlab.com www.gitlab.com
  IdentityFile ~/.ssh/gitlab

```

### 命令说明 [更多更全的命令在这](https://gist.github.com/HGladiator/65b7a1676045981248188beaeb418d28)


本地环境配置

```
Git global setup
git config --global user.name "USERNAME"
git config --global user.email "USERNAME@email.com"
```

克隆项目到本地
```
git clone ssh://git@gitlab.com:USERNAME/PROJECTNAME.git
cd test
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

本地已有文件夹
```
cd existing_folder
git init
git remote add origin ssh://git@gitlab.com:USERNAME/PROJECTNAME.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

将本地的项目推送到服务器的空项目
```
cd existing_repo
git remote rename origin old-origin
git remote add origin ssh://git@gitlab.com:USERNAME/PROJECTNAME.git
git push -u origin --all
git push -u origin --tags

```
