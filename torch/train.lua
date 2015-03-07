require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> configuring optimizer'


save = '.'
maxEpoch = 400

optimState = {
   learningRate = 0.01,
   momentum = 0.95,
   learningRateDecay = 0.00001
}
optimMethod = optim.sgd

-- Log results to files
trainLogger = optim.Logger(paths.concat(save, 'train.log'))
testLogger = optim.Logger(paths.concat(save, 'test.log'))

model:float()
criterion:float()

model:cuda()
criterion:cuda()

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
trainData.data:float()
trainData.labels:float()
testData.data:float()
testData.labels:float()
parameters, gradParameters = model:getParameters()

function train()
   shuffle = torch.randperm(trsize)
   trainData.data = trainData.data:index(1, shuffle:long())
   trainData.labels = trainData.labels:index(1, shuffle:long())


   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch

   local tloss = 0
   local correct = 0
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = trainData.data[{{t, math.min(t+batchSize-1,trainData:size())}, {}}]
      local targets = trainData.labels[{{t, math.min(t+batchSize-1,trainData:size())}, {}}]

      inputs = inputs:cuda()
      targets = targets:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

		       local ypred = model:forward(inputs)
		       local loss = criterion:forward(ypred, targets)
		       tloss = tloss + loss

		       local grad_y = criterion:backward(ypred, targets)
		       model:backward(inputs, grad_y)

		       correct = ypred:ge(0.5):eq(targets:ge(0.5)):sum()

                       -- normalize gradients and f(X)
                       gradParameters:div(batchSize)
                       f = loss

                       -- return f and df/dX
                       return f,gradParameters
      end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)


   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print("\n==> training accuracy %:")
   print(correct / trainData:size() / noutputs * 100)
   print("\n==>training loss")
   print(tloss / trainData:size())

   -- update logger/plot
   trainLogger:add{['% class accuracy (train set)'] = correct / trainData:size() / noutputs * 100, ['training loss'] = tloss / trainData:size()}

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
   local testBatchSize = 8
      -- disp progress
   for t = 1,testData:size(),testBatchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      local input = testData.data[{{t, math.min(t+testBatchSize-1,testData:size())}, {}}]
      local target = testData.labels[{{t, math.min(t+testBatchSize-1,testData:size())}, {}}]

      input = input:cuda()
      target = target:cuda()

      -- test sample
      local pred = model:forward(input)
      local loss = criterion:forward(pred, target)
      tloss = tloss + loss
      correct = pred:ge(0.5):eq(target:ge(0.5)):sum()
      -- print("\n" .. target .. "\n")

   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print('\n Test Accuracy %:')
   print(correct / testData:size() / noutputs * 100)
   print('\ntest loss:')
   print(tloss / testData:size())

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = correct / testData:size() / noutputs * 100, ['test loss'] = tloss / testData:size()}   
   -- next iteration:

end


