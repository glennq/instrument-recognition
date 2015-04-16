require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> configuring optimizer'


save = '.'
maxEpoch = 500

optimState = {
   learningRate = 0.06,
   momentum = 0.9,
   learningRateDecay = 0.0001
}
optimMethod = optim.sgd

-- Log results to files
trainLogger = optim.Logger(paths.concat(save, 'train.log'))
testLogger = optim.Logger(paths.concat(save, 'test.log'))

model:float()
criterion:float()

if opt.type == 'cuda' then 
  model:cuda()
  criterion:cuda()
end

print '==> defining training procedure'
parameters, gradParameters = model:getParameters()
tr_sum = trainData.present:sum()
te_sum = testData.present:sum()

function train()
   shuffle = torch.randperm(trsize)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   local tloss = 0
   local correct = 0
   local n_correct = 0
   local exact_correct = 0
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData.size,batchSize do
      -- disp progress
      xlua.progress(t, trainData.size)

      -- create mini batch
      local inputs = {}
      local targets = {}
      local presents = {}
      for i = t,math.min(t+batchSize-1,trainData.size) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
	 local present = trainData.present[shuffle[i]]
         if opt.type == 'cuda' then 
	    input = input:cuda() 
	    target = target:cuda()
	    present = present:cuda()
	 end
         table.insert(inputs, input)
         table.insert(targets, target)
	 table.insert(presents, present)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err
			  

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
			  local temp = output:ge(0.5):eq(targets[i]:ge(0.5))
			  correct = correct + temp:sum()
			  if temp:sum() == noutputs then
			     exact_correct = exact_correct + 1
			  end
			  n_correct = n_correct + temp:cmul(presents[i]):sum()
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
		       tloss = tloss + f
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
      end
      
      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
      
   end
   -- time taken
   tloss = tloss / trainData.size
   time = sys.clock() - time
   time = time / trainData.size

   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print("\n==> training accuracy %:")
   print(correct / trainData.size / noutputs * 100)
   print("\n==> training modified accuracy %:")
   print(n_correct / tr_sum * 100)
   print("\n==> training exact match accuracy %:")
   print(exact_correct / trainData.size * 100)
   print("\n==>training loss")
   print(tloss)

   -- update logger/plot
   trainLogger:add{['% class accuracy (train set)'] = correct / trainData.size / noutputs * 100, ['training loss'] = tloss, ['% modified accuracy (train set)'] = n_correct / tr_sum * 100, ['training exact match accuracy %'] = exact_correct / trainData.size * 100}

   -- save/log current net
   local filename = paths.concat(save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

print('==> defining test procedure')

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local tloss = 0
   local correct = 0
   local n_correct = 0
   local exact_correct = 0
   local n_tp = 0
   local n_pred = 0
   local n_pos = 0
   local f_mac_tp = torch.Tensor(noutputs):zero():float()
   local f_mac_pred = torch.Tensor(noutputs):zero():float()
   local f_mac_pos = torch.Tensor(noutputs):zero():float()

      -- disp progress
   for t = 1,testData.size do
      -- disp progress
      xlua.progress(t, testData.size)

      local input = testData.data[t]
      local target = testData.labels[t]
      local present = testData.present[t]
      if opt.type == 'cuda' then
        input = input:cuda()
        target = target:cuda()
	present = present:cuda()
      end
      -- test sample
      local pred = model:forward(input)
      local loss = criterion:forward(pred, target)
      tloss = tloss + loss
      local temp = pred:ge(0.5):eq(target:ge(0.5))
      correct = correct + temp:sum()
      if temp:sum() == noutputs then
	 exact_correct = exact_correct + 1
      end
      n_correct = n_correct + temp:cmul(present):sum()
      f_mac_tp:add(pred:ge(0.5):cmul(target:ge(0.5)):float())
      f_mac_pos:add(target:ge(0.5):float())
      f_mac_pred:add(pred:ge(0.5):float())
      n_tp = n_tp + pred:ge(0.5):cmul(target:ge(0.5)):sum()
      n_pred = n_pred + pred:ge(0.5):sum()
      n_pos = n_pos + target:ge(0.5):sum()
      -- a more agressive modified accuracy as follows:
      -- n_correct = n_correct + present:sum() - pred:ge(0.5):eq(target:le(0.5)):sum()
      -- print("\n" .. target .. "\n")

   end
   
   tloss = tloss / testData.size
   -- timing
   time = sys.clock() - time
   time = time / testData.size
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print('\n Test Accuracy %:')
   print(correct / testData.size / noutputs * 100)
   print('\ntest modified accuracy %:')
   print(n_correct / te_sum * 100)
   print('\ntest exact match accuracy %:')
   print(exact_correct / testData.size * 100)
   print('\ntest loss:')
   print(tloss)
   print('\ntest precision:')
   print(n_tp / n_pred)
   print('\ntest recall:')
   print(n_tp / n_pos)
   print('\ntest F micro')
   print(2 * n_tp / (n_pred + n_pos))
   print('\ntest F macro')
   f_mac_pos:add(f_mac_pred)
   local tt = 0
   local cnt = 0
   for i=1,(#f_mac_tp)[1] do
      if f_mac_pos[i] > 0 then
         cnt = cnt + 1
         tt = tt + f_mac_tp[i] / f_mac_pos[i]
      end
   end
   print(tt * 2 / cnt)
   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = correct / testData.size / noutputs * 100,
      ['test loss'] = tloss, ['test modified accuracy %'] = n_correct / te_sum * 100,
      ['test exact match accuracy %'] = exact_correct / testData.size * 100,
      ['test precision'] = n_tp / n_pred, ['test recall'] = n_tp / n_pos,
      ['test F micro'] = 2 * n_tp / (n_pred + n_pos), ['test F macro'] = (tt * 2 / cnt)}   
   -- next iteration:

end


