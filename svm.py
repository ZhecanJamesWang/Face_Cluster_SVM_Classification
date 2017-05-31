import pickle 
import numpy as np
from sklearn.svm import SVC
from random import shuffle
from sklearn.externals import joblib
import os
import sys
sys.path.append('/Users/zhecanwang/liblinear/python')
from liblinearutil import *

class TrainSVM(object):
	def __init__(self):
		self.inputDir = "featureData/"
		self.clusterDir = "clusterResult/"		
		self.outputDir = "svmModel/"
		self.clusterNum = 10

		self.subId2cluster = pickle.load(open( self.clusterDir + "subId2cluster.p", "rb" ) )

		self.cluster2subId = pickle.load(open( self.clusterDir + "cluster2subId.p", "rb" ) )

		self.centerInfo = pickle.load(open( self.clusterDir + "centerInfo.p", "rb" ) )
		print " type(self.centerInfo): ", type(self.centerInfo)

		self.files = os.listdir(self.inputDir)
		# print self.subId2cluster
		print "self.inputDir = featureData, len(self.files): ", len(self.files)
		print "len(self.subId2cluster.key()): ", len(self.subId2cluster.keys())		


	def train(self):
		for file in self.files:
			if file != ".DS_Store":				
				file = file.split(".")[0]
# ---------------------------------------------------------------
				# if file in self.subId2cluster:
				posCluster = int(self.subId2cluster[file])
				clusters = list(np.arange(self.clusterNum))
				print type(posCluster)
				print "posCluster: ", posCluster
				print "clusters: ", clusters

				index = clusters.index(posCluster)
				negaClusters = clusters[:index] + clusters[index + 1:]
				print "negaClusters: ", negaClusters

				posFeatures = np.load(self.inputDir + file + ".npy")
				
				negaFeatures = []
				print "self.centerInfo: ", self.centerInfo

				for cluster in negaClusters:
					# if isinstance(cluster, (int, np.int64)):
					print "int(cluster): ", cluster
					# negaCenter = self.centerInfo[int(cluster)]
					# negaFeature = np.load(self.inputDir + negaCenter + ".npy")
					path = self.clusterDir + "centerFeature/" + str(cluster) + "/"
					
					files = os.listdir(path)
					for file in files:
						if file != ".DS_Store":
							negaFeatureFile = file
							break
					negaFeature = np.load(path + negaFeatureFile)

					print "negaFeature.shape: ", negaFeature.shape
					negaFeatures.append(negaFeature)

				features = []
				features.extend(posFeatures)
				features.extend(negaFeatures)

				Np = len(posFeatures)
				Nn = len(negaFeatures)

				print "Np: ", Np
				print "Nn: ", Nn

				posLabel = [1] * Np
				negaLabel = [-1] * Nn
				label = []
				label.extend(posLabel)
				label.extend(negaLabel)

				features, label = self.shuffleList(features, label)

				features = np.asarray(features)
				label = np.asarray(label)

				C=1;

				Ep=(Np+Nn)/(2.0*Np) #Coefficient of possitive constraint iterm
				En=(Np+Nn)/(2.0*Nn) #Coefficient of negative constraint iterm

				m = train(label, features, '-s 2 -c ' + str(C) + ' -w1 ' + str(Ep) + ' -w-1 ' + str(En))

				# clf = SVC()
				# clf.fit(features, label) 
				# joblib.dump(clf, self.outputDir + file + 'svm.pkl') 

				save_model(self.outputDir + file + '.model', m)

				
# m = load_model('heart_scale.model')
# p_label, p_acc, p_val = predict(y, x, m, '-b 1')
# ACC, MSE, SCC = evaluations(y, p_label)


	def shuffleList(self, list1, list2):
		# Given list1 and list2
		list1_shuf = []
		list2_shuf = []
		index_shuf = range(len(list1))
		shuffle(index_shuf)
		for i in index_shuf:
			list1_shuf.append(list1[i])
			list2_shuf.append(list2[i])

		return list1, list2

	def selectiveTest(self):
		testFeature = np.load("testing/1.npy")
		svms = os.listdir(self.outputDir)
		for svm in svms:
			print svm
			m = load_model(self.outputDir + svm)
			p_labels, p_acc, p_vals = predict([], testFeature, m)
			print "p_labels: ", p_labels
			print "p_acc: ", p_acc
			print "p_vals: ", p_vals
			break

	def run(self):
		# self.train()
		self.selectiveTest()

if __name__ == '__main__':
	TrainSVM().run()