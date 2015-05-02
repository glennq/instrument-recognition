require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
file = '/home/glenn/data/ntt.t7'

print '==> loading dataset'

loaded = torch.load(file)
trsize = (#loaded['train_y'])[1]
tesize = (#loaded['test_y'])[1]
trainData = {
   data = loaded['train_X']:reshape(trsize, 2, 1, 44100),
   labels = loaded['train_y'],
   present = loaded['train_p'],
   size = trsize
}
testData = {
   data = loaded['test_X']:reshape(tesize, 2, 1, 44100),
   labels = loaded['test_y'],
   present = loaded['test_p'],
   size = tesize
}

loaded = nil

collectgarbage()
