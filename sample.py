import datetime as dt
import pymongo as pm
import os
import gensim as gs
import numpy as np
import gongcq.Public as Public

class Doc:
    def __init__(self, docDict):
        self.docId = docDict['_id']
        self.time = docDict['time']
        self.title = docDict['title'] + docDict['secTitle']
        self.content = docDict['content']
        self.parse = docDict['parse']
        self.tfIdfDict = {}
        self.themeIdSet = set()
        self.senList = []

    def AddThemeId(self, theme):
        self.themeIdSet.add(theme)

    def AddSen(self, sen):
        self.senList.append(sen)

class Sen:
    def __init__(self, docId):
        self.parse = []
        self.tfIdfDict = {}
        self.docId = docId
        self.seq = None
        self.themeVec = None

    def AppendWord(self, word):
        self.parse.append(word)

    def GetContent(self):
        s = ''
        for word in self.parse:
            s += word
        return s

    def Sim(self, other):
        tiList0 = []
        tiList1 = []
        intersection = self.tfIdfDict.keys() & other.tfIdfDict.keys()
        for word in intersection:
            tiList0.append(self.tfIdfDict[word])
            tiList1.append(other.tfIdfDict[word])

        # for word0, tfIdf0 in self.tfIdfDict.items():
        #     for word1, tfIdf1 in other.tfIdfDict.items():
        #         if word0 == word1:
        #             tiList0.append(tfIdf0)
        #             tiList1.append(tfIdf1)
        if len(tiList0) == 0:
            return 0
        else:
            arr0 = np.array(tiList0)
            arr1 = np.array(tiList1)
            sim = np.dot(arr0, arr1) / (np.sqrt(np.linalg.norm(arr0, ord=2)) * np.sqrt(np.linalg.norm(arr1, ord=2)))
            return sim

class Theme:
    def __init__(self, themeId):
        self.themeId = themeId
        self.docList = []
        self.docIdSet = set()
        self.vecMean = None
        self.senSimMat = None
        self.senList = []
        self.cenSort = None
        self.absDoc = None

    def AddDoc(self, doc):
        if doc.docId not in self.docIdSet:
            self.docList.append(doc)
            self.docIdSet.add(doc.docId)

    def EvalAbsDoc(self):
        '''
        evaluate the abstract document
        :return:
        '''
        # evaluate sentence similarity matrix
        self.senList = []
        for doc in self.docList:
            for sen in doc.senList:
                self.senList.append(sen)
        self.senSimMat = np.zeros([len(self.senList), len(self.senList)], dtype=float)
        for i in range(len(self.senList)):
            for j in range(i, len(self.senList)):
                sim = self.senList[i].Sim(self.senList[j])
                self.senSimMat[i][j] = sim
                self.senSimMat[j][i] = sim

        # centroid for each sentence
        self.centroidList = []
        for row in self.senSimMat:
            self.centroidList.append(np.mean(row))
        self.cenSort = np.argsort(-np.array(self.centroidList))

        # average length of documents in this cluster
        length = 0
        for doc in self.docList:
            length += len(doc.content)
        length /= (len(self.docList) if len(self.docList) > 0 else 1)

        # abstract
        self.absDoc = ''
        # --------------------------------
        # # greed
        # for i in self.cenSort:
        #     self.absDoc += self.senList[i].GetContent()
        #     if self.absDoc[-1] != '\n':
        #         self.absDoc += '\n'
        #     if len(self.absDoc) >= length:
        #         break
        # --------------------------------
        # divers
        absSenSeqSet = set()
        penaltyWeight = 0.2
        for i in self.cenSort:
            # evaluate the similarity between current sentence and the picked sentences
            pickedSimList = []
            for s in range(len(self.senList)):
                curSen = self.senList[s] # current sentence
                pickedSim = 0
                for pickedSenSeq in absSenSeqSet:
                    sim = curSen.Sim(self.senList[pickedSenSeq])
                    pickedSim += sim
                pickedSim /= max(len(absSenSeqSet), 1)
                pickedSimList.append(pickedSim)
            mmr = (1 - penaltyWeight) * np.array(self.centroidList) - \
                  penaltyWeight * np.array(pickedSimList)
            mmrSort = np.argsort(-mmr)
            for s in mmrSort:
                if s not in absSenSeqSet:
                    absSenSeqSet.add(s)
                    self.absDoc += self.senList[s].GetContent()
                    if self.absDoc[-1] != '\n':
                        self.absDoc += '\n'
                    break

            if len(self.absDoc) >= length:
                break
        # --------------------------------

class HotTheme:
    '''
    HotTheme
    '''

    #themeKeyword: a dict, the key is the name of theme,
    #and the value is a set which contains the keywords belong to the theme.
    themeKeyword = None

    # docDict: a dict, the key is the ID of document, and the value is a Doc object.
    docDict = None

    # themeDict: a dict, the key is the name of theme, and the value is a Theme object.
    themeDict = None

    # senList: a list, the elements are Sen objects which are cut from all documents
    senList = None

    def __init__(self, themeKeyword, docDict):
        '''
        construct an HotTheme object
        :param themeKeyword: a dict, the key is the name of theme,
                             and the value is a set which contains the keywords belong to the theme.
        :param docDict: a dict, the key is the ID of document, and the value is a Doc object.
        '''
        self.themeKeyword = themeKeyword
        self.docDict = docDict
        self.themeDict = {} # a dict, the key is the name of theme,
                            # and the value is a Theme object.
        for theme in self.themeKeyword.keys():
            self.themeDict[theme] = set()
        self.senList = []

        self.RoughCluster()
        self.EvalTfIdf()
        self.EvalAbsDocForTheme()

    def ClearThemeCluster(self):
        '''
        clear theme cluster information in self.themeDict, only the theme ID in self.themeDict.keys() remained
        :return:
        '''
        for themeId in self.themeDict.keys():
            self.themeDict[themeId] = Theme(themeId)

    def RoughCluster(self):
        '''
        rough clustering for initial use
        :return:
        '''
        sepSet = {'。', '\n', '；', '！', '？'}
        ignoreSet = {'\u3000'}
        self.ClearThemeCluster()
        for docId, doc in self.docDict.items():
            sen = Sen(docId)
            for w in range(len(doc.parse)):
                word = doc.parse[w]
                if word in ignoreSet:
                    continue
                # rough clustering
                if word in self.themeDict.keys():
                    self.themeDict[word].AddDoc(doc)
                    doc.AddThemeId(word)
                # cut to sentence
                sen.AppendWord(word)
                if (word in sepSet or w == len(doc.parse) - 1) and len(sen.parse) > 2:
                    doc.AddSen(sen)
                    sen = Sen(docId)

    def EvalTfIdf(self):
        '''
        evaluate tf-idf value for each document and each sentence
        :return:
        '''
        for i in [0, 1]:
            unitList = []
            unitParseList = []
            if i == 0: # for each document
                for docId, doc in self.docDict.items():
                    unitList.append(doc)
                    unitParseList.append(doc.parse)
            elif i == 1: # for each sentence
                for docId, doc in self.docDict.items():
                    for sen in doc.senList:
                        unitList.append(sen)
                        unitParseList.append(sen.parse)

            wordId = gs.corpora.Dictionary(unitParseList)
            for word, id in wordId.token2id.items():
                wordId.id2token[id] = word
            corpus = [wordId.doc2bow(unitParse) for unitParse in unitParseList]
            tfIdf = gs.models.TfidfModel(corpus)
            for u in range(len(unitList)):
                tiTupList = tfIdf[corpus[u]]
                tiDict = {}
                for tiTup in tiTupList:
                    word = wordId.id2token[tiTup[0]]
                    tiDict[word] = tiTup[1]
                unitList[u].tfIdfDict = tiDict

    def EvalAbsDocForTheme(self):
        print(str(dt.datetime.now()) + ' begin to evaluate abstract document for each theme...')
        for themeId, theme in self.themeDict.items():
            print('')
            print(str(dt.datetime.now()) + ' ' + themeId + ': document number is ' + str(len(theme.docList)))
            # if themeId == '无人机':
            if len(theme.docList) >= 2:
                theme.EvalAbsDoc()
                print(theme.absDoc)


mc = pm.MongoClient('mongodb://gongcq:gcq@192.168.5.208:27017/text')
# mc = pm.MongoClient('mongodb://gongcq:gcq@localhost:27017/text')
db = mc['text']

timeScope = [dt.datetime(2017, 11, 2), dt.datetime(2017, 11, 9)]
docs = db.section.find({'time': {'$gte': timeScope[0], '$lt': timeScope[1]}})
docDict = {}
for doc in docs:
    if doc['masterId'] != '':
        continue
    docDict[doc['_id']] = Doc(doc)

# rough clustering, and cut to text sentence.
sepSet = {'。', '\n'}
themeSet = Public.FileToSet(os.path.join('.', 'config', 'theme.txt'))
themeKeyword = {}
for themeId in themeSet:
    themeKeyword[themeId] = set()

ht = HotTheme(themeKeyword, docDict)
ddd = 0