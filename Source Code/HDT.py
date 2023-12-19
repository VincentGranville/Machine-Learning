from math import log
import time

start = time.time()

# This method updates the dictionaries based on given ID, pv and word
def update_pvs(word, pv, id, word_count_dict, word_pv_dict, min_pv_dict, max_pv_dict, ids_dict):
    if word in word_count_dict:
        word_count_dict[word] += 1
        word_pv_dict[word] += pv
        if min_pv_dict[word] > pv:
            min_pv_dict[word] = pv
        if max_pv_dict[word] < pv:
            max_pv_dict[word] = pv
        ids_dict[word].append(id)
    else:
        word_count_dict[word] = 1
        word_pv_dict[word] = pv
        min_pv_dict[word] = pv
        max_pv_dict[word] = pv
        ids_dict[word] = [id]
# dictionaries to hold count of each key words, their page views, and the ids of the article in which used.
List = dict()
list_pv = dict()
list_pv_max = dict()
list_pv_min = dict()
list_id = dict()
articleTitle = list() # Lists to hold article id wise title name and pv
articlepv = list()
sum_pv = 0
ID = 0
in_file = open("HDTdata4.txt", "r", encoding="latin-1")  # Specify the correct encoding



for line in in_file:
    if ID == 0: # excluding first line as it is header
        ID += 1
        continue
    line = line.lower()
    aux = line.split('\t') # Indexes will have: 0 - Title, 1 - URL, 2 - data and 3 - page views
    url = aux[1]
    pv = log(1 + int(aux[3]))
    if "/blogs/" in url:
        type = "BLOG"
    else:
        type = "OTHER"
#   #--- clean article titles, remove stop words
    title = aux[0]
    title = " " + title + " " # adding space at the ends to treat stop words at start, mid and end alike
    title = title.replace('"', ' ')
    title = title.replace('?', ' ? ')
    title = title.replace(':', ' ')
    title = title.replace('.', ' ')
    title = title.replace('(', ' ')
    title = title.replace(')', ' ')
    title = title.replace(',', ' ')
    title = title.replace(' a ', ' ')
    title = title.replace(' the ', ' ')
    title = title.replace(' for ', ' ')
    title = title.replace(' in ', ' ')
    title = title.replace(' and ', ' ')
    title = title.replace(' or ', ' ')
    title = title.replace(' is ', ' ')
    title = title.replace(' in ', ' ')
    title = title.replace(' are ', ' ')
    title = title.replace(' of ', ' ')
    title = title.strip()
    title = ' '.join(title.split()) # replacing multiple spaces with one
    #break down article title into keyword tokens
    aux2 = title.split(' ')
    num_words = len(aux2)
    for index in range(num_words):
        word = aux2[index].strip()
        word = word + '\t' + 'N/A' + '\t' + type
        update_pvs(word, pv, ID - 1, List,list_pv, list_pv_min, list_pv_max, list_id) # updating single words

        if (num_words - 1) > index:
            word = aux2[index] + '\t' + aux2[index+1] + '\t' + type
            update_pvs(word, pv, ID - 1, List, list_pv, list_pv_min, list_pv_max, list_id) # updating bigrams

    articleTitle.append(title)
    articlepv.append(pv)
    sum_pv += pv
    ID += 1
in_file.close()

nArticles = ID - 1  # -1 as the increments were done post loop
avg_pv = sum_pv/nArticles
articleFlag = ["NA" for n in range(nArticles)]
nidx = 0
nidx_Good = 0
nidx_Bad  = 0
pv_threshold_good = 7.1
pv_threshold_bad = 6.2
mode = 'robust method'  # options are 'perfect fit' or 'robust method'
OUT = open('hdt-out2.txt','w')
OUT2 = open('hdt-reasons.txt','w')
for idx in List:
    n = List[idx]
    Avg = list_pv[idx]/n
    Min = list_pv_min[idx]
    Max = list_pv_max[idx]
    idlist = list_id[idx]
    nidx += 1
    if mode == 'perfect fit':
      good_node = n > 2 and Min > pv_threshold_good
      bad_node  = n > 2 and Max < pv_threshold_bad
    elif mode == 'robust method': 
        # below values are chosen based on heuristics and experimenting 
        good_node = n > 3 and n < 8 and Min > 6.9 and Avg > 7.6 or \
                n >= 8 and n < 16 and Min > 6.7 and Avg > 7.4 or \
                n >= 16 and n < 200 and Min > 6.1 and Avg > 7.2
        bad_node =  n > 3 and n < 8 and Max < 6.3 and Avg < 5.4 or \
                n >= 8 and n < 16 and Max > 6.6 and Avg < 5.9 or \
                n >= 16 and n < 200 and Max > 7.2 and Avg < 6.2 
    if good_node:
        OUT.write(idx + '\t' + str(n) + '\t' + str(Avg) + '\t' + str(Min) + '\t' + str(Max) + '\t' + str(idlist) + '\n')
        nidx_Good += 1
        for ID in idlist:
            title=articleTitle[ID]
            pv = articlepv[ID]
            OUT2.write(title + '\t' + str(pv) + '\t' +  idx + '\t' + str(n) + '\t' + str(Avg) + '\t' + str(Min) + '\t' + str(Max) + '\n')
            articleFlag[ID] = "GOOD"
    elif bad_node:
        nidx_Bad += 1
        for ID in idlist:
            articleFlag[ID] = "BAD"
# Computing results based on Threshold values
pv1 = 0
pv2 = 0
n1 = 0
n2 = 0
m1 = 0
m2 = 0
FalsePositive = 0
FalseNegative = 0
for ID in range(nArticles):
    pv = articlepv[ID]
    if articleFlag[ID] == "GOOD":
        n1 += 1
        pv1 += pv
        if pv < pv_threshold_good:
            FalsePositive += 1
    elif articleFlag[ID] == "BAD":
        n2 += 1
        pv2 += pv
        if pv > pv_threshold_bad:
            FalseNegative += 1
    if pv > pv_threshold_good: 
        m1 += 1
    elif pv < pv_threshold_bad:  
        m2 += 1
#
# Printing results
avg_pv1 = pv1/n1
avg_pv2 = pv2/n2
errorRate = FalsePositive/n1
UnclassifiedRate = 1 - (n1 + n2) / nArticles
aggregationFactor = (nidx/nidx_Good)/(nArticles/n1)
print ("Average log pageview count (pv):","{0:.2f}".format(avg_pv))
print ("Avg pv, articles marked as Good:","{:.2f}".format(avg_pv1))
print ("Avg pv, articles marked as Bad :","{:.2f}".format(avg_pv2))
print()
print ("Number of articles marked as Good: ", n1, " (real number is ", m1,")", sep = "" )
print ("Number of articles marked as Bad : ", n2, " (real number is ", m2,")", sep = "")
print ("Number of false positives        :",FalsePositive,"(Bad marked as Good)")
print ("Number of false negatives        :", FalseNegative, "(Good marked as Bad)")
print ("Total number of articles         :", nArticles)
print()
print ("Proportion of False Positives: ","{0:.1%}".format(errorRate))
print ("Proportion of Unclassified   : ","{0:.1%}".format(UnclassifiedRate))
print()
print ("Aggregation factor (Good node):","{:.1f}".format(aggregationFactor))
print ("Number of feature values: ", nidx," (marked as good: ", nidx_Good,")", sep = "")
print ()
print("Execution time: ","{:.4f}".format(time.time() - start), "seconds")

