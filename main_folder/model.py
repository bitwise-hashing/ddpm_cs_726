import torch
import pytorch_lightning as pl
import math

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        self.time_embed = self.positional_encoding(n_steps,n_dim)
        self.model = NoisePredictor(n_dim,n_dim)

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed[t].squeeze(dim=1)
        
        
        return self.model(x,t_embed)

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        
        ## Linear Schedule 
        
        # self.betas = torch.linspace(lbeta,ubeta,self.n_steps)
        # self.alphas = 1-self.betas
        # self.alpha_bars = torch.cumprod(self.alphas,dim=0)
        
        # Cosine Clamp Scheduler 
        
        s = 0.008
        timesteps = (
            torch.arange(self.n_steps + 1, dtype=torch.float64) / self.n_steps + s
        )
        alphas_bars = timesteps / (1 + s) * math.pi / 2
        alphas_bars = torch.cos(alphas_bars).pow(2)
        alphas_bars = alphas_bars / alphas_bars[0]
        betas = 1 - alphas_bars[1:] / alphas_bars[:-1]
        
        betas = betas.clamp(min=lbeta,max=ubeta)
        
        # Cosine Beta Schedule
        
        # def cosine(t):
        #     return math.sin((t/self.n_steps)*math.pi/2 - math.pi/2) + 1 
        
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(cosine(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        # Quadratic Schedule
        
        # def quadratic(t):
        #     return (t/self.n_steps)**2
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(quadratic(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        #Square Root Schedule
        
        # def square_root(t):
        #     return torch.sqrt(t/self.n_steps)
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(square_root(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        self.betas = betas
        self.alphas = 1 - self.betas 
        self.alpha_bars = torch.cumprod(self.alphas,dim=0)
        

    def q_sample(self, x, t,epsilon):
        """
        Sample from q given x_t.
        """
        xt = torch.sqrt(self.alpha_bars[t])*x + torch.sqrt(1-self.alpha_bars[t])*epsilon
        return xt

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        
        n_samples = x.shape[0]
        z = torch.zeros(n_samples,self.n_dim)
            
        if t>0:
            for i in range(n_samples):
                z[i] = torch.randn((self.n_dim))
                
        with torch.no_grad():
            epsilon_predicted = self.forward(x,t)
            ep_coefficient = self.betas[t]/torch.sqrt(1-self.alpha_bars[t])
            
            x = x - ep_coefficient*epsilon_predicted
            x = x/torch.sqrt(self.alphas[t])
            
            sigma_t = torch.sqrt(self.betas[t])
            
            x = x + sigma_t*z
            
        return x

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        batch_size = batch.shape[0]
        t = torch.distributions.uniform.Uniform(0,self.n_steps).sample(torch.Size((batch_size,))).long().view(-1,1)
        epsilon = torch.zeros((batch_size,self.n_dim))
        
        for i in range(batch_size):
            epsilon[i] = torch.randn((self.n_dim))
            
        xt = self.q_sample(batch,t,epsilon)
        
        epsilon_predicted = self.forward(xt,t)
        
        loss = torch.mean(torch.square(torch.norm(epsilon-epsilon_predicted,p=2,dim=1)))
        
        return loss
        
    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        
        intermediate_results = []
        
        xt = torch.zeros(n_samples,self.n_dim)
        for i in range(n_samples):
            xt[i] = torch.randn((self.n_dim))
        
        for t in reversed(range(self.n_steps)):
            
            xt = self.p_sample(xt,t)
            intermediate_results.append(xt)
            
        if return_intermediate:
            return xt,intermediate_results
        else:
            return xt

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        self.learning_rate = 1e-3
        return torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
    
    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * torch.div(i,2,rounding_mode='floor')) / torch.tensor(d_model, dtype=torch.float32))
        return pos * angle_rates

    def positional_encoding(self,position, d_model):
        # INPUT n_steps and n_dims to the function.
        # OUTPUT 2D tensor you can query to get a time embedding of (index on t-1 as this starts from 0 to n_steps-1)
        angle_rads = self.get_angles(torch.arange(position).unsqueeze(1),
                                torch.arange(d_model).unsqueeze(0),
                                d_model)

        # apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        # Get a n_steps * n_dims tensor for positional encoding
        return angle_rads
    
class NoisePredictor(pl.LightningModule):
    def __init__(self,n_dim,embedding_shape):
        super(NoisePredictor,self).__init__()
        
        # Model with 1 Hidden Layer
        
        # self.architecture = [3,16,3]
        # self.fc1 = torch.nn.Linear(in_features=n_dim+embedding_shape,out_features=self.architecture[1])
        # self.batchnorm1 = torch.nn.BatchNorm1d(num_features=self.architecture[1])
        # self.fc2 = torch.nn.Linear(in_features=self.architecture[1]+embedding_shape,out_features=self.architecture[2])
        
        # Model with 3 Hidden Layer
        
        # self.architecture = [3,8,16,8,3]
        # self.architecture = [3,16,32,16,3]
        # self.fc1 = torch.nn.Linear(in_features=n_dim+embedding_shape,out_features=self.architecture[1])
        # self.batchnorm1 = torch.nn.BatchNorm1d(num_features=self.architecture[1])
        # self.fc2 = torch.nn.Linear(in_features=self.architecture[1]+embedding_shape,out_features=self.architecture[2])
        # self.batchnorm2 = torch.nn.BatchNorm1d(num_features=self.architecture[2])
        # self.fc3 = torch.nn.Linear(in_features=self.architecture[2]+embedding_shape,out_features=self.architecture[3])
        # self.batchnorm3 = torch.nn.BatchNorm1d(num_features=self.architecture[3])
        # self.fc4 = torch.nn.Linear(in_features=self.architecture[3]+embedding_shape,out_features=self.architecture[4])
        
        # Model with 5 Hidden Layer
        
        self.architecture = [3,16,32,64,32,16,3]
        self.fc1 = torch.nn.Linear(in_features=n_dim+embedding_shape,out_features=self.architecture[1])
        self.batchnorm1 = torch.nn.BatchNorm1d(num_features=self.architecture[1])
        self.fc2 = torch.nn.Linear(in_features=self.architecture[1]+embedding_shape,out_features=self.architecture[2])
        self.batchnorm2 = torch.nn.BatchNorm1d(num_features=self.architecture[2])
        self.fc3 = torch.nn.Linear(in_features=self.architecture[2]+embedding_shape,out_features=self.architecture[3])
        self.batchnorm3 = torch.nn.BatchNorm1d(num_features=self.architecture[3])
        self.fc4 = torch.nn.Linear(in_features=self.architecture[3]+embedding_shape,out_features=self.architecture[4])
        self.batchnorm4 = torch.nn.BatchNorm1d(num_features=self.architecture[4])
        self.fc5 = torch.nn.Linear(in_features=self.architecture[4]+embedding_shape,out_features=self.architecture[5])
        self.batchnorm5 = torch.nn.BatchNorm1d(num_features=self.architecture[5]) 
        self.fc6 = torch.nn.Linear(in_features=self.architecture[5]+embedding_shape,out_features=self.architecture[6])
        
    def forward(self,X,t):
        
        # Model with 1 Hidden Layer
        
        # model_input = torch.cat((X,t),dim=1).float()
        # out = torch.relu(self.batchnorm1(self.fc1(model_input)))
        # out = torch.cat((out,t),dim=1).float()
        # out = self.fc2(out)
        
        # Model with 3 Hidden Layer
        
        # model_input = torch.cat((X,t),dim=1).float()
        # out = torch.relu(self.batchnorm1(self.fc1(model_input)))
        # out = torch.cat((out,t),dim=1).float()
        # out = torch.relu(self.batchnorm2(self.fc2(out)))
        # out = torch.cat((out,t),dim=1).float()
        # out = torch.relu(self.batchnorm3(self.fc3(out)))
        # out = torch.cat((out,t),dim=1).float()
        # out = self.fc4(out)
        
        # Model with 5 Hidden Layer
        
        model_input = torch.cat((X,t),dim=1).float()
        out = torch.relu(self.batchnorm1(self.fc1(model_input)))
        out = torch.cat((out,t),dim=1).float()
        out = torch.relu(self.batchnorm2(self.fc2(out)))
        out = torch.cat((out,t),dim=1).float()
        out = torch.relu(self.batchnorm3(self.fc3(out)))
        out = torch.cat((out,t),dim=1).float()
        out = torch.relu(self.batchnorm4(self.fc4(out)))
        out = torch.cat((out,t),dim=1).float()
        out = torch.relu(self.batchnorm5(self.fc5(out)))
        out = torch.cat((out,t),dim=1).float()
        out = self.fc6(out)
        
        return out 
