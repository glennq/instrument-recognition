require 'cunn'

local PWECriterion, parent = torch.class('nn.PWECriterion', 'nn.Criterion')

function PWECriterion:__init()
   parent.__init(self)
end 

function PWECriterion:updateOutput(input, target)
   -- BP-MLL, a convex surrogate loss for ranking loss
   local lsize = target:sum()
   local nlsize = target:size(1) - lsize
   if lsize * nlsize == 0 then
      self.output = 0
   else
      local mat = torch.zeros(lsize, nlsize):float()
      input = input:float()
      target = target:float()
      for k = 1, mat:size(1) do
         mat[k]:add(input[target:le(0.5)])
      end
      for k = 1, mat:size(2) do
         mat[{{}, k}]:add(-1, input[target:ge(0.5)])
      end
      self.output = mat:exp():sum() / lsize / nlsize
   end
   return self.output
end

function PWECriterion:updateGradInput(input, target)
   local temp = input.new()
   local lsize = target:sum()
   local nlsize = target:size(1) - lsize
   if lsize * nlsize == 0 then
      self.gradInput = temp.new()
      self.gradInput:resizeAs(input)
      self.gradInput:fill(0)
   else
      local mat = torch.zeros(lsize, nlsize):float()
      input = input:float()
      target = target:float()
      for k = 1, mat:size(1) do
         mat[k]:add(input[target:le(0.5)])
      end
      for k = 1, mat:size(2) do
         mat[{{}, k}]:add(-1, input[target:ge(0.5)])
      end
      mat = mat:exp() / nlsize / lsize
      self.gradInput = input.new()
      self.gradInput:resizeAs(input)
      self.gradInput[target:ge(0.5)] = -mat:sum(2)
      self.gradInput[target:le(0.5)] = mat:sum(1)
      self.gradInput = self.gradInput:typeAs(temp)
   end
   return self.gradInput
end
