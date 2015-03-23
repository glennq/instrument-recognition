require 'torch'
require 'cutorch'
cutorch.setDevice(3)
print '==> processing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type','cuda','type: float | cuda')
cmd:option('-save','False','save: False | True')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'class.lua'
dofile 'train_opt.lua'

epoch = 1

while epoch < maxEpoch do
   train()
   collectgarbage()
   test()
   collectgarbage()
end
