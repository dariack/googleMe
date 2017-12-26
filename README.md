# GoogleMe

Rediscover your past months and years using Google search data.
Google allows users to download their Google data, including their search history.
The program analyzes the data and creates a personalized word-cloud poster, which brings back your thoughts, fears and dreams from that period.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

1. python version 3
```
$ sudo apt-get update
$ sudo apt-get install python3.6
```
2. pip3
```
$ sudo easy_install3 pip
```
3. Download your Google data by following the instructions in this link: https://takeout.google.com/settings/takeout and the remarks below:
- **Notice!** The only data this project needs is **Searches** data. So, it’s the only Google product that needs to be chosen
    
![screenshot2](https://user-images.githubusercontent.com/12940079/34079667-b70afe4c-e33a-11e7-9360-4afb6c7f951e.png)

- When downloading the data, keep all the **default settings**:

![screenshot22](https://user-images.githubusercontent.com/12940079/34079718-a91081a8-e33b-11e7-9db1-049df3c483c2.png)

- ***Clarification:** Your data, which obviously is very private, will not be used or transferred to any other machine during the program run. The use of the data is merely local, and nobody will have access to it but you. In other words - your private data will remain private.*

4. Unzip the downloaded data.

### Installing

1. using *requirements.txt*, pip install the required python libraries for the project
```
$ pip install -r requirements.txt
```
2. run the project
```
$ python main.py
```


## Screenshots

- Word Cloud Poster

![2017-12_wordcloud](https://user-images.githubusercontent.com/12940079/34080025-a5f5afde-e340-11e7-9c2f-60bc0c1faa79.png)

![2016-08_wordcloud](https://user-images.githubusercontent.com/12940079/34157576-e2013cf6-e4ca-11e7-8e0c-2fe852c72108.png)

- Word Trends

![watch online_trend](https://user-images.githubusercontent.com/12940079/34361162-6ed223c0-ea70-11e7-9efe-b28214bbf1e7.png)

![python_trend](https://user-images.githubusercontent.com/12940079/34361183-97598a86-ea70-11e7-959a-ca821af4103b.png)


- Number of Google queries over months

![stats1](https://user-images.githubusercontent.com/12940079/34079835-3c72f03c-e33e-11e7-9415-43b86113425b.png)

- Hebrew vs. English

![heb_eng](https://user-images.githubusercontent.com/12940079/34254999-43635a8c-e657-11e7-959f-6cd33c519f19.png)

- Gui

![gui](https://user-images.githubusercontent.com/12940079/34254963-245b91cc-e657-11e7-973d-0b04f072b387.png)


## Author

**[Daria Ackerman](https://www.linkedin.com/in/dariack/)** - *The Hebrew University Of Jerusalem*
