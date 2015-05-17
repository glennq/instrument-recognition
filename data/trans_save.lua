--[[
Read splitted mat file and save to t7 format
--]]

require 'torch'
matio = require 'matio'

data = matio.load('/scratch/jq401/ml-cqt-d3-4289821/dataset_4.mat')
torch.save('test.t7', data)
data = nil
collectgarbage()
print('finished test')
data1 = matio.load('/scratch/jq401/ml-cqt-d3-4289821/dataset_0.mat')
data2 = matio.load('/scratch/jq401/ml-cqt-d3-4289821/dataset_1.mat')
data3 = matio.load('/scratch/jq401/ml-cqt-d3-4289821/dataset_2.mat')
data4 = matio.load('/scratch/jq401/ml-cqt-d3-4289821/dataset_3.mat')

print('finished reading train')
for k, v in pairs(data1) do
   data1[k] = v:cat(data2[k], 1)
   data2[k] = nil
   collectgarbage()
   data1[k] = data1[k]:cat(data3[k], 1)
   data3[k] = nil
   collectgarbage()
   data1[k] = data1[k]:cat(data4[k], 1)
   data4[k] = nil
   collectgarbage()
   print('finished '..k)
end
torch.save('train.t7', data1)


