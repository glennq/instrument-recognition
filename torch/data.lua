require 'torch'

file = '/home/glenn/data/tt.t7'

print '==> loading dataset'

loaded = torch.load(file)
trsize = (#loaded['train_y'])[1]
tesize = (#loaded['test_y'])[1]
trainData = {
   data = loaded['train_X']:contiguous():view(trsize, 2, 44100):transpose(2,3),
   labels = loaded['train_y'],
   size = function() return trsize end
}
testData = {
   data = loaded['test_X']:contiguous():view(tesize, 2, 44100):transpose(2,3),
   labels = loaded['test_y'],
   size = function() return tesize end
}

loaded = nil

collectgarbage()
