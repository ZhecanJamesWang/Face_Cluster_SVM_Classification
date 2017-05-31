import os

class DimenReduce(object):
	def __init__(self):
		self.inputDir = "data/"
		self.outputDir = "redeuced/"
		self.outputDim = None
		files = os.listdir(self.inputDir)
		
	def pca(self)：

	def run(self):
		self.pca()

if __name__ == '__main__':
	DimenReduce().run()