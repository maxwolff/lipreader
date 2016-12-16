# Parses files, formats data into dictionaries. called in segProblem.py

#------ returns array of array of sentences 

def makeSentenceArr(corpusFileName):
    parseSentences = []
    with open(corpusFileName) as f:
        sentence = f.readlines()
        for i in range(len(sentence)):
            reduce1 = sentence[i].split()
            redLen = len(reduce1)
            reduce1 = reduce1[1:redLen]
            parseSentences.append(reduce1)
    return parseSentences

# ------- returns array of arrays of the given phoneme translations from spoken sentences. 

def makePhoArr(fileName):
    firstArr = []
    counter = 0 
    with open(fileName) as f:
        sentence = f.readlines()
        sentenceNum = len(sentence)
        for i in range(sentenceNum):
            firstWords = sentence[i].split()
            firstArr.append(firstWords[2:-1])

    return firstArr


# -------  creates dic of all phoneme translations for all words by combining time-encoded word and phoneme files 

def parseSenTime(corpusFileName):
    parseSentences = []
    with open(corpusFileName) as f:
        sentence = f.readlines()
        for i in range(len(sentence)):
            sentenceTimes = []
            reduce1 = sentence[i].split()
            reduce1 = reduce1[1:]
            for i in range(len(reduce1)/3): 
                index1 = i*3
                index2 = i*3 + 3
                select = reduce1[index1:index2]
                sentenceTimes.append(select)
            parseSentences.append(sentenceTimes)
    return parseSentences

def parsePhoTime(corpusFileName):
    parseSentences = []
    with open(corpusFileName) as f:
        sentence = f.readlines()
        for i in range(len(sentence)):
            sentenceTimes = []
            reduce1 = sentence[i].split()
            reduce1 = reduce1[4:]
            for i in range(len(reduce1)/3 - 1): 
                index1 = i*3
                index2 = i*3 + 3
                select = reduce1[index1:index2]
                sentenceTimes.append(select)
            parseSentences.append(sentenceTimes)
    return parseSentences

def makePhonemeDic(wordTimes, phoTimes):
    dic = {}
    for i in range(len(wordTimes)): 
        for word in wordTimes[i]: 
            arr = []
            transArr = []
            for pho in phoTimes[i]:
                if int(pho[1]) >= int(word[1]) and int(pho[2]) <= int(word[2]): 
                    arr.append(pho[0])
            if word[0] in dic: 
                spellings = dic[word[0]]
                newSpelling = True
                for entryNum in range(len(spellings)): 
                    spelling = spellings[entryNum]
                    if arr == spelling[0]: 
                        newSpelling = False
                        num = spelling[1] + 1
                        dic[word[0]][entryNum][1] = num
                if newSpelling: 
                    dic[word[0]].append([arr,1]) 
            else: 
                newArr = []
                newArr.append([arr,1])
                dic[word[0]] = newArr
    return dic

# ----- returns dic that takes a phoneme array and returns the number of occurances of different translations for those phonemes
# ------ eg: dic{ ['iy', 'dh'] : [[dog: 3], [cat: 5]]}

def makeTranslationCosts(transDic):
    newDic = {}
    for k,v in transDic.iteritems(): 
        arr = transDic[k]
        for item in arr: 
            tupledEntry = tuple(item[0])
            if tupledEntry in newDic:
                oldEntries = newDic[tupledEntry]
                newEntry = True
                for i in range(len(oldEntries)): 
                    if oldEntries[i][0] == k: 
                        # add
                        newDic[tupledEntry][i][1] += 1
                        newEntry = False
                if newEntry: 
                    arr = newDic[tupledEntry]
                    arr.append([k, 1])
                    newDic[tupledEntry] = arr
            else: # word not in dic
                newArr = []
                newArr.append([k,item[1]])
                newDic[tupledEntry] = newArr
    return newDic

#----- returns dic of phoneme array to (words, number of occurences)

def makeWordCosts(senFile, phoFile):
    wordTimes = parseSenTime(senFile) # ['word', startTime, endTime]
    phoTimes = parsePhoTime(phoFile) # ['pho', startTime, endTime]
    transDic = makePhonemeDic(wordTimes, phoTimes) # {'word': [['pho1', 'pho2']:#occurances], [['pho1', 'pho2']:#occ]}
    wordCosts = makeTranslationCosts(transDic) # {('pho1', 'pho2'): [['word', # occ]}
    return wordCosts

# ----- returns bigram costs. { 'big' : [ [dog:4], [cat: 7] ]}

def makeBigramCostDic(sentences):
    bigramDic = {}
    with open(sentences) as f:
        sentence = f.readlines()
        sentenceNum = len(sentence)
        for i in range(sentenceNum):
            words = sentence[i].split()
            words = words[1:]
            for j in range(len(words)-1): 
                # repeat word. is it a new bigram?
                if words[j] in bigramDic: 
                    # go through all existing entries for this word 
                    newEntry = True
                    for entryNum in range(len(bigramDic[words[j]])): 
                        if bigramDic[words[j]][entryNum][0] == words[j + 1]: 
                            freq = bigramDic[words[j]][entryNum][1]
                            bigramDic[words[j]][entryNum][1] = freq + 1
                            newEntry = False
                    if newEntry: 
                        newArr = bigramDic[words[j]]
                        newArr.append([words[j+1],1])
                        bigramDic[words[j]] = newArr
                else: 
                    newArr = []
                    newArr.append([words[j+1],1])
                    bigramDic[words[j]] = newArr
    return bigramDic

def makeBigramCosts(fileName):
    return makeBigramCostDic(fileName)


# --------- returns firstWord (percent of occurances at the beginning of sentence 
# eg: { 'big' : .03 }

# first word of a sentence : num occurnences
def makeFirstDict(fileName):
    firstDict = {}
    with open(fileName) as f:
        sentence = f.readlines()
        sentenceNum = len(sentence)
        for i in range(sentenceNum):
            freq = 0
            firstWords = sentence[i].split()
            for i in range(1,len(firstWords)): 
                if firstWords[1] in firstDict:
                    freq = firstDict[firstWords[1]]
                firstDict[firstWords[1]] = freq + 1           
    return firstDict 
    
# first word of a sentence : decimal
def makeProbDict(occDict):
    sumOcc = 0
    for k,v in occDict.iteritems():
        sumOcc += v
    probDict = {}
    for k,v in occDict.iteritems():
        probDict[k] = occDict[k] * 1.0 / sumOcc
    return probDict
 
def makeFirstCosts(fileName):    
    firstDict = makeFirstDict(fileName)
    return makeProbDict(firstDict)
