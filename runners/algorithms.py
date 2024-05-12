import torch
import torch.nn as nn
import torch.optim as optim

class XmorphLearning:
    def __init__(self, 
                 Xmorph_net,
                 device='cpu',
                 num_learning_epochs=1,
                num_mini_batches=1,
                learning_rate=1e-4,
                **kwargs):
        
        self.device = device
        self.Xmorph_net = Xmorph_net.to(device=device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.Xmorph_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        
    def init_storage(self, num_envs, num_storage, input_shape, output_shape,):
        self.storage = SLstorage( num_storage,num_envs, input_shape, output_shape, self.device)

    def store(self, input, tgt):
        self.storage.add_data(input.detach(), tgt.detach())

    def update(self,):
        mean_reconst_loss =  0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for input_batch, tgt_batch in generator:
            
            scr2tgt = self.Xmorph_net.forward(input_batch)
            forward_loss = self.loss_func(scr2tgt, tgt_batch)
            tgt2scr = self.Xmorph_net.inverse(tgt_batch)
            inverse_loss = self.loss_func(tgt2scr, input_batch)
            loss = forward_loss+inverse_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
          
            mean_reconst_loss += loss.item()


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_reconst_loss /= num_updates
        self.storage.clear()
        return mean_reconst_loss
        


class SLstorage():
    def __init__(self,
                 num_storage, num_envs, input_shape, output_shape, device='cpu') -> None:
        self.num_storage = num_storage
        self.num_envs = num_envs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device


        self.inputs = torch.zeros(( num_storage, num_envs, *input_shape), dtype=torch.float32, device=device)
        self.tgts = torch.zeros((  num_storage, num_envs,*output_shape), dtype=torch.float32, device=device)

        self.clear()


    def add_data(self, input, tgt):
        self.inputs[self.step].copy_(input)
        self.tgts[self.step].copy_(tgt)
        self.step = (self.step + 1) % self.num_storage

    def clear(self):
        self.step = 0
    

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_storage
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)


        inputs = self.inputs.flatten(0, 1)
        tgts = self.tgts.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                input_batch = inputs[batch_idx]
                tgt_batch = tgts[batch_idx]

                yield input_batch, tgt_batch