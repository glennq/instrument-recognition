require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

noutputs = 82
batchSize = 1

-- input dimensions
nfeats = 2
length = 44100
ninputs = nfeats*length

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,400}
filtsize = {401, 301}
poolsize = {10, 10}
stridesize = {10, 10}
viewsize = 407

print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[1],1, stridesize[1], 1))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[2],1, stridesize[1], 1))

-- stage 3 : 
model:add(nn.Reshape(viewsize*nstates[2]))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*viewsize, nstates[3]))
model:add(nn.ReLU())

-- stage 4:
model:add(nn.Linear(nstates[3], noutputs))
model:add(nn.Sigmoid())

-- loss:
criterion = nn.BCECriterion()


