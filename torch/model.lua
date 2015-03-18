require 'torch'   -- torch
require 'cunn'      -- provides all sorts of trainable modules/layers

noutputs = 82
batchSize = 32

-- input dimensions
nfeats = 2
length = 44100
ninputs = nfeats*length

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,400}
filtsize = {401, 435}
poolsize = {16, 12}
stridesize = {8, 6}
viewsize = 836

print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.TemporalConvolution(nfeats, nstates[1], filtsize[1], 1))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(poolsize[1],stridesize[1]))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.TemporalConvolution(nstates[1], nstates[2], filtsize[2], 1))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(poolsize[2],stridesize[2]))

-- stage 3 : 
model:add(nn.Reshape(viewsize*nstates[2], true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*viewsize, nstates[3]))
model:add(nn.ReLU())

-- stage 4:
model:add(nn.Linear(nstates[3], noutputs))
model:add(nn.Sigmoid())

-- loss:
criterion = nn.BCECriterion()


