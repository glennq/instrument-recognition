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

dofile '/home/jq401/ml_project/code8_raw_temporal/data.lua'
--dofile '/home/jq401/ml_project/code5_large_temporal_BCE/PWECriterion.lua'
dofile '/home/jq401/ml_project/code8_raw_temporal/model.lua'
dofile '/home/jq401/ml_project/code8_raw_temporal/train.lua'

epoch = 1

while epoch < maxEpoch do
   train()
   collectgarbage()
   test()
   collectgarbage()
end
