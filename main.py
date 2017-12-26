#!/usr/bin/env python3
import random
import easygui as gui
import sys

import matplotlib.image as mpimg
from operator import itemgetter, attrgetter

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
plt.rcParams['keymap.save'] = ''

import numpy as np
import statistics
import pandas as pd
import os
import json
import datetime
import gzip
import pickle
import re
import progressbar

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from nltk import ngrams
import nltk

import pprint
from nltk.stem.lancaster import LancasterStemmer

# from __future__ import division
# from sklearn.cluster import KMeans
# from numbers import Number
# from pandas import DataFrame
# import sys, codecs, numpy

nltk.download('stopwords')

# -*- coding: utf-8 -*-




path = "./"



"""
unzip the output of google's downloaded data
"""
def unzip():
    curr_path = path + "/Takeout/Searches"

    # assuming the dir is already unzipped!
    for filename in os.listdir(curr_path):
        parse_json(curr_path + "/" + filename)
        return

    '''
    for filename in os.listdir(curr_path):
        # print(filename)
        if os.path.isdir(filename):
            if filename == "Takeout":
                print("# no need to unzip")
                parse_json(curr_path)
                return

    # if got here, need to unzip:
    for filename in os.listdir(curr_path):
        if os.path.isdir(filename):
            if filename != "Takeout":
                print("# !!!need to unzip!!!: " + filename)
                # tar = gzip.open(filename)
                # tar     #TODO unzip the files!
                # tar.close()
    '''


def extract_query_text(item):

    try:
        query_text = item["query"]["query_text"]
        return (query_text)
    except:
        print("!! ERROR - no query text")
        raise


def extract_query_time(item):
    try:
        query_time_lst = item["query"]["id"]
        query_timestamp = query_time_lst[0]["timestamp_usec"]
        temp = int(query_timestamp) / 1000000.0
        query_time = datetime.datetime.fromtimestamp(temp).strftime("%Y-%m-%d %H:%M:%S")

        # get month and year from date:
        query_time_obj = datetime.datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S").date()
        curr_year = '%02d' % query_time_obj.year
        curr_month = '%02d' % query_time_obj.month

        return (query_time, curr_month, curr_year)

    except:
        print("!! ERROR - no query time")
        raise


def get_start_month_from_file_name(filename):
    filename = filename[:-5]
    filename = filename.split(" ")
    year = filename[-1]
    end_month = filename[-2][:3]
    end_month_num = datetime.datetime.strptime(end_month, '%b').month
    end_month_num = '%02d' % end_month_num

    # return the end month because in the query files - the first query we will read in the string is the latest

    return (end_month_num, year)


def month_to_ignore(curr_month):
    if int(curr_month) == 12:
        return 1
    else:
        return int(curr_month) + 1

def get_pickle_name(year, month, special=""):
    return "queries_" + str(year) + "-" + str(month) + special + ".pickle"


def save_monthly_queries(year, month, queries_lst):

    if not os.path.exists("pickle_dump/"):
        os.makedirs("pickle_dump/")

    # 1. save all the queries of the month in a single file:

    # print("# total queries for " + str(month) + "-" + str(year) + " are: " + str(len(queries_lst)))
    with open("pickle_dump/" + get_pickle_name(year, month), 'wb') as fp:
        pickle.dump(queries_lst, fp)

    # 2. split the queries into two files:
        # A) geo queries - when checking the directions from A to B in google maps
        # B) "classic" google queries

    geo_q = []
    classic_q = []

    for query in queries_lst:
        query_text = query['query_text']
        if (" -> " in query_text) or (" <- " in query_text):
            geo_q.append(query)
        else:
            classic_q.append(query)

    # 3. write to files:

    with open("pickle_dump/" + get_pickle_name(year, month, ".geo"), 'wb') as fp:
        pickle.dump(geo_q, fp)

    with open("pickle_dump/" + get_pickle_name(year, month, ".classic"), 'wb') as fp:
        pickle.dump(classic_q, fp)

    # 4. return geo vs. classic statistics:

    num_of_geo = len(geo_q)
    try:
        percentage_of_geo = (len(geo_q)/len(queries_lst)*100)
    except:
        percentage_of_geo = 0

    return num_of_geo, percentage_of_geo

"""
extract all the queries from the json files in the path,
parse them into pairs of query_text and query_time,
and save the results as one json file compressed in gzip
"""
def parse_json(path):

    this_dir = path

    if this_dir.endswith("Takeout"):
        curr_path = this_dir + "/Searches"
    elif "Takeout" in os.listdir(this_dir):
        curr_path = this_dir + "/Takeout/Searches"
    else:
        # something wring with given path (can't find "Takeout"), maybe it's in the code's dir:
        this_dir = os.getcwd()
        if this_dir.endswith("Takeout"):
            curr_path = this_dir + "/Searches"
        elif "Takeout" in os.listdir(this_dir):
            curr_path = this_dir + "/Takeout/Searches"
        else:
            print("!! Error - please make sure the path you chose ends with the folder 'Takeout'") 
            raise Exception

    month_statistics = {}       # number of searches per month

    l1 = os.listdir(curr_path)
    l1.sort()

    pbar = progressbar.ProgressBar()

    for filename in pbar(l1):
        query_lst_month = []  # all queries of curr month

        f = open(curr_path + "/" + filename, encoding='utf-8')

        curr_month, curr_year = get_start_month_from_file_name(filename)
        ignore_month = month_to_ignore(curr_month)

        data = json.load(f)
        data = data["event"]


        for item in data:

            temp_dict = {}          # will hold the current query text and date

            query_time, q_month, q_year = extract_query_time(item)
            temp_dict["query_time"] = query_time

            if ignore_month == int(q_month):
                continue

            if curr_year != q_year:
                print("!! Error - Something is wrong with the query dates")
                print("curr_year = " + str(curr_year))
                print("q_year = " + str(q_year))

            query_text = extract_query_text(item)
            temp_dict["query_text"] = query_text


            # if we need to switch months:
            if curr_month != q_month:


                # save all queries of this month and update file:
                save_monthly_queries(curr_year, curr_month, query_lst_month)
                month_statistics[str(curr_year) + "-" + str(curr_month)] = len(query_lst_month)
                curr_month = q_month

                # reset month list:
                query_lst_month = []

            query_lst_month.append(temp_dict)
            # print(temp_dict)


        # save all queries of this month and update file:
        save_monthly_queries(curr_year, curr_month, query_lst_month)
        month_statistics[str(curr_year) + "-" + str(curr_month)] = len(query_lst_month)

    return month_statistics

"""
returns the number of english queries from the given month, and the total amount of queries
"""
def statistic_monthly_english_vs_hebrew(month, year):
    # TODO update!!
    f = gzip.open("queries_" + year + "-" + month + ".json.gz", "rb")
    data = pickle.load(f)

    total_queries = len(data)
    english_queries = 0

    for item in data:
        if is_english(item["query_text"]) == True:
            english_queries = english_queries + 1

    return total_queries, english_queries

"""
returns a dictionary: the key is a string %year-%month ("2016-02"),
and the value is tuple: (total_queries, english_queries)
"""
def statistic_wide_hebrew_vs_english():

    res_dict = {}

    curr_year = 2007
    curr_month = 8
    while True:
        try:
            total_q, english_q = statistic_monthly_english_vs_hebrew(str("%02d" % curr_month), str(curr_year))
        except:
            # print("!! no data for " + str("%02d" % curr_month) + "-" + str(curr_year))
            total_q = 0
            english_q = 0

        if (curr_month == 12):
            curr_year = curr_year + 1
            curr_month = 1
        elif (curr_year == 2017 and curr_month == 11):
            break
        else:
            curr_month = curr_month + 1

        res_dict[str(curr_year) + "-" + str("%02d" % curr_month)] = (total_q, english_q)

    return res_dict

"""
checks if a given string is in english, returns True / False
"""
def is_english(s):
    try:
        ans = detect(s)
    except:
        return False
    if ans == "en":
        return True
    return False

"""
returns True if a string contains a hebrew char
"""
def is_hebrew(s):
    try:
        ans = detect(s)
    except:
        return False
    if ans == "he":
        return True
    return False



def calc_stats(stats):

    vals = [x[1] for x in stats]

    # calc mean:
    my_mean = statistics.mean(vals)

    # calc median:
    my_median = statistics.median(vals)

    return my_mean, my_median

def generate_plots(stats):

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(111)
    # ax.axhline(y=n, label='Old')
    # ax.plot([5, 6, 7, 8], [100, 110, 115, 150], 'ro', label='New')

    x = (range(len(stats)))
    y = [x[1] for x in stats]
        # stats.values()

    plt.bar(x, y, align='center', color='grey')

    x_lables = []

    last_year = [x[0].split("-")[0] for x in stats]
    # last_year = list(stats.keys())[0].split("-")[0]

    for obj in stats:
        obj = obj[0]
        curr_year = obj.split("-")[0]
        if curr_year != last_year:
            x_lables.append(obj)
            last_year = curr_year
        else:
            month = obj.split("-")[1]
            x_lables.append(month)

    if len(x_lables) == 0:
        print("x_lables is zero!!!")

    plt.xticks(range(len(stats)), x_lables)#, FontSize=7)
    plt.xticks(rotation=90)

    ax = plt.axes()
    ax.yaxis.grid(linestyle='dotted')  # horizontal lines


    plt.ylabel('Number Of Queries')
    plt.xlabel('Months')

    my_mean, my_median = calc_stats(stats)

    plt.axhline(y=my_mean, linewidth=0.5, color='r')#xmin=2,xmax
    plt.axhline(y=my_median, linewidth=0.5, color='g')


    ax.annotate('means', xy=(0, my_mean), xycoords='data',
                horizontalalignment='left', verticalalignment='top',
                color='r')#, FontSize=10, FontWeight='bold')

    ax.annotate('median', xy=(0, my_median), xycoords='data',
                horizontalalignment='left', verticalalignment='top',
                color='g')#, FontSize=10, FontWeight='bold')


    plt.title("Number Of Google Queries Over Months")

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(13, 8)
    figure.subplots_adjust(bottom=0.15, left=0.08)

    if not os.path.exists("figs/"):
        os.makedirs("figs/")

    # when saving, specify the DPI
    plt.savefig("figs/stats.png", dpi=300)

    plt.show()


def random_font_color():

    # Colormaps colors:
    color_options = ['Pastel2', 'Set3', 'Paired', 'Set2', 'Pastel1', 'hsv', 'PiYG', 'RdYlBu', 'RdYlGn', 'Spectral',
                     'coolwarm', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'BuPu',
                     'GnBu', 'PuBu', 'rainbow', 'PuBuGn', 'BuGn', 'YlGn', 'spring', 'summer',
                     'autumn', 'cool', 'Wistia', 'viridis']

    choice = random.choice(color_options)
    return(choice)


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def generate_heb_pie_chart(year, month):

    try:
        filename = "queries_{}-{}.classic.pickle".format(year, month)

        data = pickle.load(open("pickle_dump/" + filename, 'rb'))
        data_queries = [word['query_text'] for word in data]

        # hebrew stats
        heb_queries = [is_hebrew(q) for q in data_queries]
        num_heb = heb_queries.count(True)

        # other lang stats
        num_other = len(data_queries) - num_heb

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Hebrew', 'English'
        sizes = [num_heb, num_other]
        explode = (0.03, 0.03)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Hebrew vs. English Queries\n{}-{}".format(year, month))
        plt.show()

    except:
        raise Exception


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    '''
    Return the top n features that on average are most important amongst documents in rows
    indentified by indices in grp_ids.
    :param Xtr: the sparse matrix
    :param features:
    :param grp_ids:
    :param min_tfidf:
    :param top_n:
    :return: the top n features
    '''

    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def get_filter_words(words_to_filter=[]):
    '''
    Generate a list of words to filter out, including stopwords
    :param words_to_filter: a list of words to filter. default = []
    :return: a list of words to filter out
    '''

    filter_words_lst = set(stopwords.words('english'))

    # to save the embarrassment...
    # filter_words_lst.add("סקס")
    # filter_words_lst.add("פורנו")
    # filter_words_lst.add("sex")
    # filter_words_lst.add("porn")
    # filter_words_lst.add("fuck")

    for word in words_to_filter:
        print("# add " + word + " to filter")
        filter_words_lst.add(word)

    return list(filter_words_lst)


def sub_words_helper(query_list, n=3):
    '''
    Returns the n-1-grams to remove to the given list
    :param query_list: a list of strings
    :param n: n-grams
    :return: the sub-grams to remove
    '''

    to_remove = []
    for i in range(1, n + 1):
        temp_i_lst = []

        for query in query_list:
            n1 = list(ngrams(re.split('\W', query), i))
            # for tup in n1:
            for bigram_tuple in n1:
                if len(bigram_tuple) == 3:
                    x = "%s %s %s" % bigram_tuple
                if len(bigram_tuple) == 2:
                    x = "%s %s" % bigram_tuple
                elif len(bigram_tuple) == 1:
                    x = "%s" % bigram_tuple
                temp_i_lst.append(x)

        to_remove.append(temp_i_lst)
    return to_remove


def remove_sub_words_from_n_grams(query_list, n=3):
    '''
    The function filters out sub-grams.
    For example: if the given query list contains the query "new york city" (= trigram),
    then, the sub-grams are removed: "new york" "york city" (= bigrams) and "new", "york", "city (= unigrams)
    :param query_list: a list of strings
    :param n: the max n-grams this query list contains
    :return: a query list after filter
    '''

    lst = sub_words_helper(query_list, n)
    trigrams = lst[2]
    bigrams = lst[1]
    unigrams = lst[0]

    res = sub_words_helper(trigrams, n=2)
    bi_remove = res[1]
    uni_remove1 = res[0]

    bi_after_filter = [word for word in bigrams if word not in bi_remove]

    res = sub_words_helper(bi_after_filter, n=1)

    uni_remove2 = res[0]

    uni_after_filter = [word for word in unigrams if ((word not in uni_remove1) and (word not in uni_remove2))]

    queries_after_filter = trigrams + bi_after_filter + uni_after_filter

    return queries_after_filter


def generate_corpus_for_range(year, month, range_in_months, go_back_the_full_range_from_given_date):
    '''
    The function opens the needed pickle files, and gathers the queries into one big corpus
    :param year: string "yyyy"
    :param month: string "mm"
    :param range_in_months: positive int
    :param go_back_the_full_range_from_given_date: boolean - if True, we go from the given date, the full range backwards
                                                            if False, the date will be in the middle of the range
    :return: the corpus of the range
    '''

    if range_in_months < 1:
        print("!! ERROR - range in months must be larger than zero")
        return

    # calculate what year and month should we start from:
    if go_back_the_full_range_from_given_date:
        range_in_years = int((range_in_months) / 12)
        go_back_in_months = int((range_in_months) % 12)
    else:
        range_in_years = int((range_in_months/2) / 12)
        go_back_in_months = int((range_in_months/2) % 12)

    curr_year = int(year) - range_in_years
    curr_month = int(month) - go_back_in_months

    if (curr_month < 1):
        curr_month = 12 + curr_month
        curr_year = curr_year - 1

    # extract the queries from the needed months:
    corpus = []

    for i in range(range_in_months):

        try:
            with open("pickle_dump/" + get_pickle_name(str(curr_year), '%02d' % curr_month, ".classic"), 'rb') as fp:
                data = pickle.load(fp)  # data is a list of dics: [{'query_text': '...', 'query_time': '...'}, {...}, ...]

            for query in data:
                text = query['query_text']

                corpus.append(text)

        except:
            continue

        # continue to next month

        if curr_month == 12:
            curr_month = 1
            curr_year = curr_year + 1
        else:
            curr_month = curr_month + 1

    return corpus


def get_n_most_frequent_queries(year, month, range_in_months, n, go_back_the_full_range_from_given_date, filter_lst=[]):
    '''
    return the n most frequent queries for a given date in a given range
    :param year: string "yyyy"
    :param month: string "mm"
    :param range_in_months: positive int
    :param n: number of queries
    :param start_from_given_month_yaer: boolean - if True, we go from the given date, the full range backwards
                                                if False, the date will be in the middle of the range
    :return: a DataFrame with the n top phrases and its frequencies
    '''

    corpus = generate_corpus_for_range(year, month, range_in_months, go_back_the_full_range_from_given_date)

    # generate an engine that will vectorize the corpus, minding the filter words and 1-to-3-grams
    vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words=get_filter_words(filter_lst))

    try:
        X = vectorizer.fit_transform(corpus)        # X is a sparse matrix representing the corpus, after vectorizing it

        features = vectorizer.get_feature_names()

        top_n_phrases = (top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=n))
        # top = list(top_n_phrases.to_dict()['feature'].values())

    except:
        raise EnvironmentError

    return top_n_phrases



def generate_wordcloud(year, month, range_in_months=1, n=150, start_from_given_month_yaer=False, filter_lst = [], to_show=False):
    '''
    Generate wordcloud for a given date
    :param year: string: "yyyy"
    :param month: string "mm"
    :param range_in_months: int - generate the wordcloud for queries made in the given range
    :param n: int - maximum words in the wordcloud poster
    :param start_from_given_month_yaer: boolean - if True, we go from the given date, the full range backwards
                                                if False, the date will be in the middle of the range
    :param filter_lst: a list of words to filter out from the wordcloud
    :param to_show: boolean - if False, the wordcloud would not appear on the screen, only saved in a directory
    '''

    try:
        top_queries = (get_n_most_frequent_queries(
            year, month, range_in_months, n, start_from_given_month_yaer, filter_lst=filter_lst))

    except EnvironmentError:
        raise EnvironmentError

    top_queries_lst = list(top_queries.to_dict()['feature'].values())

    after_filter = remove_sub_words_from_n_grams(top_queries_lst, n=3)

    df = top_queries[top_queries.feature.isin(after_filter)]

    d = df.to_dict()
    words_lst = (list(d['feature'].values()))
    rate_lst = (list(d['tfidf'].values()))

    # convert to the dict format needed for the wordcloud: {<query>: <frequency>}

    wordcloud_dict = {}
    for i in range(len(words_lst)):
        if is_hebrew(words_lst[i]):
            words_lst[i] = words_lst[i][::-1]
        wordcloud_dict[words_lst[i]] = rate_lst[i]

    # Generate a word cloud image
    try:
        wordcloud = WordCloud(font_path='Alef-Regular.ttf', width=800, height=400, scale=2, background_color="black",
                              colormap=random_font_color(), stopwords=get_filter_words(filter_lst),
                              collocations=True).generate_from_frequencies(wordcloud_dict)


        cloud_name = str(year) + "-" + str(month) + "_wordcloud.png"
        tmp_path = "/tmp/figs/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        # save wordcloud to file
        wordcloud.to_file(tmp_path + cloud_name)


        img = Image.open(tmp_path + cloud_name)
        img_with_border = ImageOps.expand(img, border=55, fill='white')
        img_with_border.save(tmp_path + "border_" + cloud_name)

        poster_header = "Your wordcloud poster for " + str(year) + "-" + str(month)


        img = Image.open(tmp_path + "border_" + cloud_name)
        width, height = img.size
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype("Alef-Regular.ttf", 32)

        w, h = draw.textsize(poster_header, font=font)
        draw.text(((width - w) / 2, 10), poster_header, fill="black", font=font)

        if not os.path.exists("figs/"):
            os.makedirs("figs/")

        img.save("figs/" + cloud_name)

        if to_show:

            img = mpimg.imread("figs/" + cloud_name)
            imgplot = plt.imshow(img, interpolation='bilinear')
            plt.axis("off")
            plt.show()


    except ValueError:
        print("!! ERROR with queries at " + str(year) + "-" + str(month) + ", so no wordcloud produced")


def create_wordcloud_for_all_months(filter_lst):
    '''
    Generate wordcloud for all available months
    '''

    directory = os.fsencode(path + "/pickle_dump")

    i = 0
    total = len(os.listdir(directory))
    for file in os.listdir(directory):

        i += 1

        if i%10 == 0:
            print("# processing {}/{} months".format(i,total))

        filename = os.fsdecode(file)

        if filename.endswith(".classic.pickle"):
            filename_lst = filename[8:15].split("-")
            year = filename_lst[0]
            month = filename_lst[1]

            try:
                generate_wordcloud(year, month, filter_lst=filter_lst)
                continue
            except:
                continue

        else:
            continue

def check_if_word_in_cluster(word):

    # print("word to cluster" + word)
    f = open("glove_clusters_30000_words.json", 'r')
    data = json.load(f)

    for cluster in data.values():
        if word in cluster:
            return cluster

    return []

def run_trends(word, stemming = True):

    word = word.lower()
    print("word is: " + word + "\nstemming status is: " + str(stemming))

    st = LancasterStemmer()

    if get_trend(word, stemming) == False:
        return False # no qureies

    cluster = check_if_word_in_cluster(word)

    # try stemming if not found in cluster in regular form
    if (cluster) == 0:
        cluster = check_if_word_in_cluster(st.stem(word))

    if len(cluster) == 0:
        print("empty cluster for: " + word)
    else:
        pprint.pprint(cluster)

def get_phrase_appearance(word, stemming):
    st = LancasterStemmer()
    orig_phrases = set()
    dict = {}
    stopWords = set(stopwords.words('english'))
    for fn in os.listdir('./pickle_dump'):
        if (not fn.startswith('queries')) or (not fn.endswith(".classic.pickle")):
            continue

        data = pickle.load(open("pickle_dump/" + fn, 'rb'))

        for q in data:
            mytime = datetime.datetime.strptime(q["query_time"], "%Y-%m-%d %H:%M:%S")
            text = q["query_text"].lower().split()

            year = mytime.isocalendar()[0]
            month = mytime.month
            week = (mytime.isocalendar()[1] // month)

            if stemming:
                stem_text = [st.stem(x) for x in text]

                if len(word.split()) == 1:

                    # stemming word and text
                    word = st.stem(word)
                    # print("after stemming:" + str(word))

                    if word in stem_text:

                        # add original phrase to the set for later:
                        orig_phrases.add(text[stem_text.index(word)])
                        if text[stem_text.index(word)] not in orig_phrases:
                            print("added "+ str(text[stem_text.index(word)]) + " to set")

                        dict[year] = dict.get(year,{})
                        dict[year][month] = dict[year].get(month, {})
                        dict[year][month][week] = dict[year][month].get(week,0) + 1

                elif len(word.split()) == 2:

                    # stemming word and text
                    word1 = st.stem(word.split()[0])
                    word2 = st.stem(word.split()[1])
                    # print("after stemming:" + str(word1) + " " + str(word2))

                    if word1 in stem_text and word2 in stem_text:
                        # add original phrase to the set for later:
                        orig_phrases.add(text[stem_text.index(word1)])
                        orig_phrases.add(text[stem_text.index(word2)])

                        if text[stem_text.index(word1)] not in orig_phrases:
                            print("added " + str(text[stem_text.index(word1)]) + " to set")
                        if text[stem_text.index(word2)] not in orig_phrases:
                            print("added " + str(text[stem_text.index(word2)]) + " to set")

                        dict[year] = dict.get(year, {})
                        dict[year][month] = dict[year].get(month, {})
                        dict[year][month][week] = dict[year][month].get(week, 0) + 1

                else:
                    print("Error! please insert up to a 2 words phrase")
                    raise Exception


            else: # no stemming

                if len(word.split()) == 1:

                    if word in text:

                        dict[year] = dict.get(year, {})
                        dict[year][month] = dict[year].get(month, {})
                        dict[year][month][week] = dict[year][month].get(week, 0) + 1


                elif len(word.split()) == 2:

                    # stemming word and text
                    word1 = st.stem(word.split()[0])
                    word2 = st.stem(word.split()[1])
                    # print("after stemming:" + str(word1) + " " + str(word2))

                    if word1 in stem_text and word2 in stem_text:
                        # add original phrase to the set for later:
                        orig_phrases.add(text[stem_text.index(word1)])
                        orig_phrases.add(text[stem_text.index(word2)])

                        if text[stem_text.index(word1)] not in orig_phrases:
                            print("added "+ str(text[stem_text.index(word1)]) + " to set")
                        if text[stem_text.index(word2)] not in orig_phrases:
                            print("added "+ str(text[stem_text.index(word2)]) + " to set")

                        dict[year] = dict.get(year,{})
                        dict[year][month] = dict[year].get(month, {})
                        dict[year][month][week] = dict[year][month].get(week,0) + 1

                else:
                    print("Error! please insert up to a 2 words phrase")
                    raise Exception

    return dict, orig_phrases

def get_trend(word, stemming):

    dict, orig_phrases = get_phrase_appearance(word, stemming)

    if len(dict) == 0:
        print("No queries found for the phrase: " + str(word))
        return False

    first_year = list(dict.keys())[0]
    last_year = list(dict.keys())[-1]

    for year in range(first_year,last_year+1):
        if year not in dict.keys():
            dict[year] = {1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{},11:{},12:{}}   # empty year
        for i in range(1,13):
            if i not in dict[year].keys():
                dict[year][i] = {1:0,2:0,3:0,4:0}
            for week in range(1,5):
                if week not in dict[year][i].keys():
                    dict[year][i][week] = 0

    dict_sorted = sorted(dict.items())
    # pprint.pprint(dict_sorted)

    # dict to plot
    mydict = {}


    for year in dict_sorted:
        str_year = year[0]
        for mon in year[1].keys():
            week = year[1][mon]
            for w in week.keys():
                mydict[(str_year, mon, w)] = week[w]

    dict_sorted = sorted(mydict.items(), key=itemgetter(0,1))
    xlables = [str(x[0]) for x in dict_sorted]
    yvals = [x[1] for x in dict_sorted]

    m = max(yvals)

    for i in range(len(yvals)):
        if yvals[i] != 0:
            currlable = str(xlables[i][1:-1]).split(",")
            currlable = str(currlable[0:2]).replace("[", "")
            currlable = str(currlable).replace("]", "")
            currlable = str(currlable).replace("'", "")
            plt.text(x=i, y=yvals[i] + 0.12*m, s=currlable, size=6,verticalalignment='top', horizontalalignment='left', rotation=70)

    plt.bar(range(len(yvals)), yvals , align='center', color='grey')

    ax = plt.axes()
    ax.get_xaxis().set_visible(False)
    ax.set_ylim([0, 130*m/100])
    ax.yaxis.grid(linestyle='dotted')  # horizontal lines


    plt.ylabel('Number Of Queries (per week)')
    plt.xlabel('Time (year, week)')

    if is_hebrew(word):
        #reverse the words fot the plot:
        word = word[::-1]

    plt.title("Trend of '"+word+"' in Google Queries Over Weeks")

    if len(orig_phrases) != 0:
        all_phrases = ""
        for x in orig_phrases:
            if is_hebrew(word):
                x = x[::-1]
            all_phrases += ", " + x
        all_phrases = all_phrases[2:]
        plt.figtext(0.02, .95, 'Phrases included in the analysis:\n' + all_phrases, fontsize=9, ha='left', color="green")

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(13, 7)
    figure.subplots_adjust(bottom=0.1, left=0.08)

    if not os.path.exists("figs/"):
        os.makedirs("figs/")

    # clean word before saving, to use in file name
    word = word.replace("\"","")
    word = word.replace("\'","")
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace(";","")

    plt.savefig("figs/"+word+"_trend.png", dpi=300)

    plt.show()


def gui_welcome(title):
    # Window #1 - welcome

    ret_val = gui.msgbox(
        "Welcome to GoogleMe!\n\nBefore we get started, make sure you read the README file and followed the instructions."
        "\n\nEspecially, if you have yet to download your google search history, as explained in the README, please do that before you continue.\n\n",
        title=title, ok_button="I'm Ready!", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_ask_to_init(title):
    msg = "--Initialize The Program?--\n\n\nIf it's your first time running the program, or if you would like to re-initialize your data - you can do so now."
    ans = gui.ccbox(msg, title, choices=("It's my first time", "I come here often"), image=None,
          default_choice="It's my first time")     # show a Continue/Cancel dialog

    if ans == None:
        sys.exit(0)
    else:
        return ans           # if ans is true then need init

def gui_find_path(title):
    ret_val = gui.msgbox(
        "--Find Data's Path--\n\n\nTo get started, let's find the directory in which your google data is waiting\n"
        "Attention! The path needs to end with the 'Takeout' folder",
        title=title, ok_button="Let's!", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_get_path(title):
    # Window #2 - find path

    msg = "Where is your data?"
    data_path = gui.diropenbox(msg, title)
    if data_path is None:
        sys.exit(0)

    # make sure that none of the fields were left blank
    while 1:
        if (data_path.strip() == ""):
            errmsg = "Path is required!"
        else:
            break
        data_path = gui.enterbox(errmsg, title)
        if data_path is None:
            break

    return data_path

def gui_extracting_data(title):
    # Window #3 - wait while processing

    ret_val = gui.msgbox("We'll begin extracting your raw google data.\n\nThis may take a few seconds...\nHold tight.",
                         title=title, ok_button="Extract!", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_wait_for_all_month_poster(title):

    ret_val = gui.msgbox("We'll begin generating posters for all dates!\n\nThis may take a while...\n",
                         title=title, ok_button="I'm Patient", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_finish_for_all_month_poster(title):

    cwd = os.getcwd()
    path = cwd + "/figs"
    ret_val = gui.msgbox("All Done!\n\nYou can find your posters in:\n" + path,
                         title=title, ok_button="Awesome", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_choose_action(title):
    # Window #4 - stats or data?

    msg = "--GoogleMe!--\n\n\nWhat would you like to check out?"

    choices = ["Data Poster", "Cross Months Statistics", "Hebrew vs. English Statistics", "Word Trend"]
    choice = gui.choicebox(msg, title, choices)

    if choice is None:  # User closed msgbox
        sys.exit(0)

    return choice, choices

def gui_poster_to_pick_dates(title):
    msg = "--Poster--\n\n\n" \
          "You could go all-in and generate posters for the entire data,\n" \
          "or pick specific dates for the poster,\n" \
          "or just go wild and show a poster of a random date"
    choices = ("All In", "I'm Picky", "Surprise me", "-Add Filter-")
    ans = gui.indexbox(msg, title, choices=choices, image=None, default_choice="Surprise me")

    if ans == None:
        sys.exit(0)

    else:
        return ans

def gui_choose_dates(title, choices):
    msg = "Choose dates:\n"

    choice = gui.multchoicebox(msg, title, choices, preselect=None)

    if choice is None:  # User closed msgbox
        sys.exit(0)

    return choice

def gui_error_no_queries_selected_date(title):

    ret_val = gui.msgbox("Oh no!\n\n\nIt seems there are not enough queries in the date chosen.\n"
                         "Sorry about that.\n\nWhy don't you try picking different dates?\n",
                         title=title, ok_button="I'll pick different dates", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_error_hebrew_vs_english(title):

    ret_val = gui.msgbox("Oh no!\n\n\nIt seems there is something wrong with this action.\n"
                         "Sorry about that.\n\nWhy don't you try something else?\n",
                         title=title, ok_button="Sure, no worries", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)

def gui_filter_words(title):
    ans = gui.textbox(msg="--Filter Out Words?--\n\n\nThe poster will display google searches you have made in the past."
                          "\nInsert words you would like to filter out from the poster (or leave empty)\n\n"
                          "The words should be seperated with a comma, case insensitive\n"
                          "As long as the program is running, the filtered words are saved\n\n"
                          "Anyways, don't worry!\nOnly you can see the poster :)",title=title,
                      text="<word1>, <word2>, <word3>, ...",
                      codebox=False, callback=None, run=True)
    return ans

def gui_word_trend(title):
    ans = gui.textbox(msg="--Trends--\n\n\nCheck out the number of searches of a given word or phrase (up to a 2 words phrase)",title=title,
                      text="",
                      codebox=False, callback=None, run=True)
    return ans


def init_program(title):
    try:
        # get data path:
        gui_find_path(title)
        data_path = gui_get_path(title)

        gui_extracting_data(title)

        # extracting:
        month_statistics = parse_json(data_path)

        stat_lst = []

        keylist = list(month_statistics.keys())
        keylist.sort()
        for key in keylist:
            stat_lst.append((key, month_statistics[key]))

        all_months = sorted(list(month_statistics.keys()))
        pickle.dump(all_months, open("all_months", 'wb'))
        pickle.dump(stat_lst, open("month_statistics", 'wb'))

    except:
        raise Exception

def check_program_was_init(title):
    curr_path = os.getcwd()

    if "pickle_dump" not in os.listdir(curr_path):
        gui_need_init(title)
        return False

    return True

def gui_need_init(title):
    ret_val = gui.msgbox("Wait a sec!\n\n\nIt seems you have yet to initialize the program...\n\n"
                         "You almost got me.\n",
                         title=title, ok_button="I'm sorry. I'll be good from now on", image=None)
    if ret_val is None:  # User closed msgbox
        sys.exit(0)


def main_loop(title):
    '''
    The main loop of the gui, showing the user the available actions
    :param title: string - the name of the program
    '''

    # get the list of all dates that has data:
    all_months = pickle.load(open("all_months", "rb"))

    # a list of filtered words:
    filters = []

    ans, choices = gui_choose_action(title)

    poster = choices[0]
    all_time_stats = choices[1]
    heb_eng_stats = choices[2]
    trend = choices[3]

    if ans == poster:

        ans2 = gui_poster_to_pick_dates(title)

        poster_for_all_dates = 0
        poster_for_specific_dates = 1
        poster_for_rand_dates = 2

        ask_for_filter = 3

        if ans2 == poster_for_all_dates:  # extract all posters
            gui_wait_for_all_month_poster(title)
            create_wordcloud_for_all_months(filter_lst=filters)
            gui_finish_for_all_month_poster(title)

        elif ans2 == poster_for_specific_dates:
            dates_chosen = gui_choose_dates(title, choices=all_months)

            for date in dates_chosen:
                date_lst = date.split("-")
                year = date_lst[0]
                month = date_lst[1]

                try:
                    generate_wordcloud(year, month, to_show=True, filter_lst=filters)

                except EnvironmentError:
                    gui_error_no_queries_selected_date(title)
                    continue

        elif ans2 == poster_for_rand_dates:
            date = random.choice(all_months)
            date_lst = date.split("-")
            year = date_lst[0]
            month = date_lst[1]
            # try:
            generate_wordcloud(year, month, to_show=True, filter_lst=filters)

            # except EnvironmentError:
            #     print(EnvironmentError)
            #     gui_error_no_queries_selected_date(title)

        elif ans2 == ask_for_filter:
            # filter words
            words_to_filter = gui_filter_words(title)
            if words_to_filter == "<word1>, <word2>, <word3>, ...":
                words_to_filter = ""

            if words_to_filter != "":
                words_to_filter = words_to_filter.strip().split(",")
                words_to_filter = [word.strip() for word in words_to_filter if word.strip() != ""]
                filters = filters + words_to_filter


    elif ans == all_time_stats:
        month_statistics = pickle.load(open("month_statistics", "rb"))
        generate_plots(month_statistics)


    elif ans == heb_eng_stats:
        dates_chosen = gui_choose_dates(title, choices=all_months)

        for date in dates_chosen:
            date_lst = date.split("-")
            year = date_lst[0]
            month = date_lst[1]

            try:
                generate_heb_pie_chart(year, month)

            except Exception:
                gui_error_hebrew_vs_english(title)
                continue

    elif ans == trend:
        phrase = gui_word_trend(title)
        run_trends(phrase)


def main():
    '''
    # get_trend("running")
    run_trends('ת"א')

    '''

    global path
    # path = "/home/daria/Documents/private/seminar"
    title = "GoogleMe"

    # https://takeout.google.com/settings/takeout

    # unzip()

    gui_welcome(title)
    need_init = gui_ask_to_init(title)

    if need_init:
        init_program(title)

    else:
        if not check_program_was_init(title):
            init_program(title)
        check_program_was_init(title)
    while True:
        main_loop(title)



main()
