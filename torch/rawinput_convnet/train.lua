require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> configuring optimizer'


save = '.'
maxEpoch = 400

optimState = {
   learningRate = 0.05,
   momentum = 0.95,
   learningRateDecay = 0.00001
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

classes = {'Main System',
 'accordion',
 'acoustic guitar',
 'alto saxophone',
 'auxiliary percussion',
 'bamboo flute',
 'banjo',
 'baritone saxophone',
 'bass clarinet',
 'bass drum',
 'bassoon',
 'bongo',
 'brass section',
 'cello',
 'cello section',
 'chimes',
 'claps',
 'clarinet',
 'clarinet section',
 'clean electric guitar',
 'cymbal',
 'darbuka',
 'distorted electric guitar',
 'dizi',
 'double bass',
 'doumbek',
 'drum machine',
 'drum set',
 'electric bass',
 'electric piano',
 'erhu',
 'female singer',
 'flute',
 'flute section',
 'french horn',
 'french horn section',
 'fx/processed sound',
 'glockenspiel',
 'gong',
 'gu',
 'guzheng',
 'harmonica',
 'harp',
 'horn section',
 'kick drum',
 'lap steel guitar',
 'liuqin',
 'male rapper',
 'male singer',
 'male speaker',
 'mandolin',
 'melodica',
 'oboe',
 'oud',
 'piano',
 'piccolo',
 'sampler',
 'scratches',
 'shaker',
 'snare drum',
 'soprano saxophone',
 'string section',
 'synthesizer',
 'tabla',
 'tack piano',
 'tambourine',
 'tenor saxophone',
 'timpani',
 'toms',
 'trombone',
 'trombone section',
 'trumpet',
 'trumpet section',
 'tuba',
 'vibraphone',
 'viola',
 'viola section',
 'violin',
 'violin section',
 'vocalists',
 'yangqin',
 'zhongruan'}

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
   print("\n==>training loss")
   print(tloss)

   -- update logger/plot
   trainLogger:add{['% class accuracy (train set)'] = correct / trainData.size / noutputs * 100, ['training loss'] = tloss, ['% modified accuracy (train set)'] = n_correct / tr_sum * 100}

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
      n_correct = n_correct + temp:cmul(present):sum()
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
   print('\ntest loss:')
   print(tloss)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = correct / testData.size / noutputs * 100, ['test loss'] = tloss, ['test modified accuracy %'] = n_correct / te_sum * 100}   
   -- next iteration:

end


