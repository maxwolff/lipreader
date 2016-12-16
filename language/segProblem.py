import util
import makeDics


# would include firstWord, bigram, and occurance costs in algorithm
# but phoneme accuracy too low for sentence construction anyway
# as is, accuracy = 66%


# create search problem for UCS
class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, firstCosts, bigramCost, wordCost):
        self.wordCost = wordCost
        self.bigramCost = bigramCost
        self.firstCosts = firstCosts
        self.queryLen = len(query)
        self.query = query
        self.sentence = ""
    def startState(self):
        return (0, "")
    def isEnd(self, state):
        if self.queryLen == state[0]: 
            self.sentence = state[1]
            return True
        else: 
            return False
    def succAndCost(self, state):
        returnList = []
        lastWord = None
        if state[1]: 
            lastWord = state[1].split()
            if len(lastWord) > 1:         
                lastWord = lastWord[-1]
        for i in xrange(self.queryLen - state[0]):
            newStr = self.query[(state[0]): (state[0] + 1 + i)] 
            if tuple(newStr) in wordCosts:
                choice = wordCosts[tuple(newStr)][0][0]
                #self.totalCost(lastWord, choice)
                returnList.append((newStr, (state[0] + 1 + i, state[1] + choice + " " ), 2))
                # cost = 2 for correctly identified word. 
                
            else: 
                if i < 5: #  max length for an unknown word
                    returnList.append((newStr, (state[0] + 1 + i, state[1] + " ??? "), 10)) 
                    # cost = 15 if we couldnt find a word. allows for lossy phoneme data
        return returnList

# for one sentence: send to 
def segmentWords(query, firstCosts, bigramCost, wordCost):
    if len(query) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, firstCosts, bigramCost, wordCost))
    return ucs.result.split()

def getAccuracy(firstCosts, bigramCosts, wordCosts, sentences,phoSentences, iters): 
    counter = 0 
    right = 0
    for i in range(len(sentences)): 
        sentence = sentences[i]
        phonemeSentence = phoSentences[i]
        prediction = segmentWords(phonemeSentence, firstCosts, bigramCosts, wordCosts)
        for i in range(len(sentence)):
            if len(prediction) > i: 
                if prediction[i] == sentence[i]: 
                    right += 1
            counter += 1
        if counter > iters: # get accuracy with num iterations
            return (right , counter)
    
senTimeFile = "senTime.txt"
phoTimeFile = "phoneTime.txt"
sentenceFile = "wordSentences.txt"
phonemeFile = "phoSentences.txt"

wordCosts = makeDics.makeWordCosts(senTimeFile, phoTimeFile)
bigramCosts = makeDics.makeBigramCosts(sentenceFile)
makeFirstCosts = makeDics.makeFirstCosts(sentenceFile)

sentences = makeDics.makeSentenceArr(sentenceFile)
phonemes = makeDics.makePhoArr(phonemeFile)

iterations = 200 

accuracy = getAccuracy(makeFirstCosts, bigramCosts, wordCosts, sentences, phonemes, iterations)

print "correct:", accuracy[0], " out of: ", accuracy[1]
