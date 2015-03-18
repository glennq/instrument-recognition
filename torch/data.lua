require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
file = 'tt.t7'

print '==> loading dataset'

loaded = torch.load(file)
trsize = (#loaded['train_y'])[1]
tesize = (#loaded['test_y'])[1]
trainData = {
   data = loaded['train_X']:reshape(trsize, 2, 44100):transpose(2,3),
   labels = loaded['train_y'],
   size = trsize
}
testData = {
   data = loaded['test_X']:reshape(tesize, 2, 44100):transpose(2,3),
   labels = loaded['test_y'],
   size = tesize
}

loaded = nil

collectgarbage()
