require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

noutputs = 28
batchSize = 16

-- input dimensions
nfeats = nfeats or 84
length = 87
ninputs = nfeats*length

-- hidden units, filter sizes (for ConvNet only):
nstates = {128,128,128,400}
filtsize = {4, 5, 4}
filtsize2 = {4, 3, 3} 
poolsize = {2, 2, 2}
stridesize = {2, 1, 1}
viewsize = 33

print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(1, nstates[1], filtsize[1], filtsize2[1]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[1], poolsize[1], stridesize[1], stridesize[1]))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize2[2]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[2], poolsize[2], stridesize[2], stridesize[2]))

-- stage 3 :
model:add(nn.SpatialConvolution(nstates[2], nstates[3], filtsize[3], filtsize[3]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[3], poolsize[3], stridesize[3], stridesize[3]))

-- stage 3 : 
model:add(nn.Reshape(viewsize*viewsize*nstates[3]))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[3]*viewsize*viewsize, nstates[4]))
model:add(nn.ReLU())

-- stage 4:
model:add(nn.Linear(nstates[4], noutputs))
model:add(nn.Sigmoid())

-- loss:
criterion = nn.BCECriterion()

