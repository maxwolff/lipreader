import tensorflow as tf
import cnn_build
import shell
import util

class SegmentationProblem(util.SearchProblem):
    def __init__(self, vid, unigramCost):
        self.vid = vid
        self.unigramCost = unigramCost
        self.queryLen = len(query)
    def startState(self):
        return 0
    def isEnd(self, state):
        return bool(self.queryLen == state)
    def succAndCost(self, state):
        returnList = []
        for i in xrange(self.queryLen - state):
            newStr = self.query[(state): (state + 1 + i)]
            returnList.append((newStr, state + 1 + i, self.unigramCost(newStr)))
        return returnList
def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))
    return  " ".join(ucs.actions) 