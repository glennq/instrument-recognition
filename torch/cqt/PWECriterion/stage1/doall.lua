require 'torch'
print '==> processing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type','float','type: float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

print '==> executing all'

dofile '/home/jq401/ml_project/code1/data.lua'
dofile '/home/jq401/ml_project/code1/PWECriterion.lua'
dofile '/home/jq401/ml_project/code1/model.lua'
dofile '/home/jq401/ml_project/code1/train.lua'

epoch = 1

while epoch < maxEpoch do
   train()
   collectgarbage()
   test()
   collectgarbage()
end
