import torch
import torch.nn as nn
import torch.optim as optim

class CycLearning:
    def __init__(self,
                 h_reconst_net,
                 r_reconst_net,
                 learning_rate=1e-3,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 schedule="fixed",
                 device='cpu',
                 **kwargs) -> None:
        
        self.device = device
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        # SupervisedLearning components
        self.h_reconst_net = h_reconst_net
        self.h_reconst_net.to(self.device)
        self.h_optimizer = optim.Adam(self.h_reconst_net.parameters(), lr=learning_rate)
        self.r_reconst_net = r_reconst_net
        self.r_reconst_net.to(self.device)
        self.r_optimizer = optim.Adam(self.r_reconst_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.step = 0


    def init_storage(self,  num_envs, num_storage, input_shape, output_shape,):
        self.storage = SLstorage( num_storage,num_envs, input_shape, output_shape, self.device)

    def store(self, input, tgt):
        self.storage.add_data(input.detach(), tgt.detach())

    def update(self,):
        mean_h_reconst_loss, mean_r_reconst_loss = 0, 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for input_batch, tgt_batch in generator:
            h_prediction = self.h_reconst_net(input_batch)
            h_loss = self.loss(h_prediction, tgt_batch)
            self.h_optimizer.zero_grad()
            h_loss.backward()
            self.h_optimizer.step()
            mean_h_reconst_loss += h_loss.item()

            r_prediction = self.r_reconst_net(h_prediction.detach())
            r_loss = self.loss(r_prediction, input_batch)
            self.r_optimizer.zero_grad()
            r_loss.backward()
            self.r_optimizer.step()
            mean_r_reconst_loss += r_loss.item()


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_h_reconst_loss /= num_updates
        mean_r_reconst_loss /= num_updates
        self.storage.clear()
        return mean_h_reconst_loss, mean_r_reconst_loss
        



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