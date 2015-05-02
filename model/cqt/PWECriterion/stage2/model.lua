require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

noutputs = 11
batchSize = 16

-- input dimensions
nfeats = nfeats or 180
length = 87
ninputs = nfeats*length

-- hidden units, filter sizes (for ConvNet only):
nstates = {96,128,128,400}
filtsize = {4, 5, 4}
filtsize2 = {31, 21, 13} 
poolsize = {2, 2, 2}
stridesize = {2, 2, 2}
poolsize2 = {4, 4, 4}
stridesize2 = {2, 2, 2}
viewsize = 19
viewsize2 = 26

print '==> construct model'

model = torch.load('/scratch/jq401/run-3883064/model_back.net')

model_thres = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model_thres:add(nn.SpatialConvolution(1, nstates[1], filtsize[1], filtsize2[1]))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[1], poolsize2[1], stridesize[1], stridesize2[1]))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model_thres:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize2[2]))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[2], poolsize2[2], stridesize[2], stridesize2[2]))
--[[
-- stage 3 :
model_thres:add(nn.SpatialConvolution(nstates[2], nstates[3], filtsize[3], filtsize2[3]))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[3], poolsize2[3], stridesize[3], stridesize2[3]))
--]]
-- stage 3 : 
model_thres:add(nn.Reshape(viewsize*viewsize2*nstates[3]))
model_thres:add(nn.Dropout(0.5))
model_thres:add(nn.Linear(nstates[3]*viewsize*viewsize2, nstates[4]))
model_thres:add(nn.ReLU())

-- stage 4:
model_thres:add(nn.Linear(nstates[4], 1))
model_thres:add(nn.Sigmoid())

-- loss:
criterion = nn.AbsCriterion()

