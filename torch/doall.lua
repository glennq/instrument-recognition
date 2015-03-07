require 'torch'
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'train.lua'

epoch = 1

while epoch < maxEpoch do
   train()
   collectgarbage()
   test()
   collectgarbage()
end
