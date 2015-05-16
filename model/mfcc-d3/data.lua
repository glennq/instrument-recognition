require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
tr_file = 'train.t7'
te_file = 'test.t7'

print '==> loading dataset'
print('==> training file: '..tr_file)
print('==> test file: '..te_file)

nfeats = 60

loaded1 = torch.load(tr_file)
trsize = (#loaded1['y'])[1]
trainData = {
   data = loaded1['X']:reshape(trsize, nfeats, 1, 87),
   labels = loaded1['y'],
   present = loaded1['p'],
   size = trsize
}

loaded1 = nil

collectgarbage()

loaded2 = torch.load(te_file)
tesize = (#loaded2['y'])[1]
testData = {
   data = loaded2['X']:reshape(tesize, nfeats, 1, 87),
   labels = loaded2['y'],
   present = loaded2['p'],
   size = tesize
}

loaded2 = nil

collectgarbage()

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i = 1,nfeats do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i = 1,nfeats do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

for i = 1,nfeats do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..i..'-channel, mean: ' .. trainMean)
   print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..i..'-channel, mean: ' .. testMean)
   print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end
