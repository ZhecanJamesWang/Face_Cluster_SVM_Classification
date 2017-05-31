from sklearn.cluster import KMeans
import numpy as np
import os
import cv2
import pickle
import sys

class Clustering(object):
	def __init__(self):
		self.rawInputDir = "data/"
		self.inputDir = "featureData/"
		self.outputDir = "clusterResult/"

		self.rawFiles = os.listdir(self.rawInputDir)
		self.files = os.listdir(self.inputDir)

		self.features = []
		self.subId = []

		self.clusterNum = 10
		self.subId2cluster = {}
		self.cluster2subId = {}
		self.centerInfo = {}

		print "len(files): ", len(self.rawFiles)
		print "len(files): ", len(self.files)
		print "self.files[:10]: ", self.files[:10]

		self.testing = False

	def loadFeature(self):

		for file in self.files:
			# if file == "1.npy":
			# 	raise "debug 1.py"

			if file != ".DS_Store":
				feature = np.load(self.inputDir + file)
				subId = file.split(".")[0]
				# print "subId: ", subId
				# print "file: ", file
				# print "file.split('.'')[0]: ", subId
				# print "feature.shape: ", feature.shape
				feature = np.mean(feature, axis = 0)
				# print "after meaing .......... feature.shape: ", feature.shape
				self.features.append(feature)
				self.subId.append(subId)
		self.features = np.array(self.features)
		
		print "self.features.shape: ", self.features.shape
		
		print "len(self.subId): ", len(self.subId)
		print "len(self.features): ", len(self.features)

		print "len(subId) == len(self.features): ", len(self.subId) == len(self.features)
		if len(self.subId) != len(self.features):
			raise "debug len(subIdList) != len(imgList)"

	def KmeansCluster(self):
		self.X = np.array(self.features)
		self.kmeans = KMeans( n_clusters=self.clusterNum, random_state=0).fit(self.X)
		self.labels = self.kmeans.labels_

	def ClusterIndicesNumpy(self, clustNum, labels_array): #numpy 
	    return np.where(labels_array == clustNum)[0]

	def evalCluster(self):
		# find center person
		print "save center person ----------------------"
		for i in range(self.clusterNum):
			print "cluster: ", i
			d = self.kmeans.transform(self.X)[:, i]
			ind = np.argsort(d)[::][:1]
			subId = self.subId[ind]
			print "subId: ", subId
			# print "X[ind].shape: ", self.X[ind].shape
			centerFile = subId
			self.centerInfo[i] = centerFile

			imgName = os.listdir(self.rawInputDir + centerFile)[0]
			img = cv2.imread(self.rawInputDir + centerFile + "/" + imgName, 1)
			path = self.outputDir + "center/" + str(i)
			if not os.path.isdir(path):
				os.mkdir(path) 
			if not self.testing:
				cv2.imwrite(path + "/" + subId + ".jpg",img)


			index = self.ClusterIndicesNumpy(i, self.labels)
			print "len(index): ", len(index)
			clusterFeatures = self.X[index]
			print "clusterFeatures.shape: ", clusterFeatures.shape
			centerFeature = np.mean(clusterFeatures, axis = 0)
			print "centerFeature.shape: ", centerFeature.shape
			
			path = self.outputDir + "centerFeature/" + str(i)
			if not os.path.isdir(path):
				os.mkdir(path) 
			if not self.testing:
				np.save(path + "/" + "centerFeature" + str(i) + ".npy", centerFeature)

		if not self.testing:
			pickle.dump( self.centerInfo, open( self.outputDir + "centerInfo.p", "wb" ) )

		# eval clustering of all data	
		print "save other clustered data ---------------------"
		print "len(self.labels): ", len(self.labels)
		print "len(self.rawFiles): ", len(self.rawFiles)
		for file in self.files:
			file = file.split(".")[0]
			if file != ".DS_Store":
				try:

					index = self.files.index(file + '.npy')
					subId = self.subId[index]
					cluster = self.labels[index]

					self.subId2cluster[subId] = cluster
					if cluster in self.cluster2subId:
						self.cluster2subId[cluster].append(subId)
					else:
						self.subId2cluster[cluster] = [subId]

					imgNames = os.listdir(self.rawInputDir + file)
					for imgName in imgNames:
						if imgName != ".DS_Store":
							img = cv2.imread(self.rawInputDir + file + "/" + imgName, 1)
							path = self.outputDir + "labeling/" + str(cluster)
							if not os.path.isdir(path):
								os.mkdir(path) 
							if not self.testing:
								cv2.imwrite(path + "/" + subId + ".jpg",img)
							break

				except Exception as e:
					print "e: ", e
					# raise "debug"
		if not self.testing:
			pickle.dump( self.subId2cluster, open( self.outputDir + "subId2cluster.p", "wb" ) )
			pickle.dump( self.cluster2subId, open( self.outputDir + "cluster2subId.p", "wb" ) )
	
	def testCluster(self):
		pass
		# sd = '102'
		# index = self.subId.index(sd)
		# print kmeans.predict(self.X[sd])

	def run(self):
		self.loadFeature()
		self.KmeansCluster()
		self.evalCluster()
		self.testCluster()


if __name__ == '__main__':
	Clustering().run()


