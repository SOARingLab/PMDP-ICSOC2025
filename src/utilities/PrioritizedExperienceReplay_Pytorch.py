import heapq
import torch

class PrioritizedReplayBuffer:
    def __init__(
        self,
        max_samples=10000,
        batch_size=64,
        rank_based=False,
        alpha=0.5,
        beta0=0.4,
        beta_rate=0.99992,
        device = 'cuda',

        beta_start=0.4,
        beta_end = 0.8,
        beta_number = 30000
       

        #beta_end=1.0, 
        #beta_steps=6000
        
    ):
        self.max_samples = max_samples
        self.memory = []
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based  # if not rank_based, then proportional
        self.alpha = alpha  # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0  # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0  # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.EPS = 1e-6
        self.explore_prob = 0.1
        self.device = device

        self.beta_index = 0
        self.betas = self.generate_beta_schedule(beta_start, beta_end, beta_number)
        self.beta_end = beta_end
        '''
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.current_step = 0

        # Calculate the exponential growth rate for beta
        self.beta_rate = (self.beta_end / self.beta_start) ** (1 / self.beta_steps)
        '''

    
    def generate_beta_schedule(self, beta_start, beta_end, n, beta_min=0.00001, beta_max=1.0):

    
        """
        Generates a list of n beta values, each updated exponentially.

        Parameters:
        - beta_start (float): initial beta value.
        - beta_end (float): final beta value.
        - n (int): number of elements to generate.
        - beta_min (float): minimum beta value.
        - beta_max (float): maximum beta value.

        Returns:
        - beta_schedule (list): list of n beta values.
        """
        beta_schedule = []
        beta = beta_start
        beta_rate = (beta_end / beta_start) ** (1 / n)  # Automatically calculate decay rate
        for _ in range(n):
            beta = min(beta_max, max(beta_min, beta * beta_rate))
            beta_schedule.append(beta)
        return beta_schedule
    
    '''

    def generate_beta_schedule(self, beta_start, beta_end, n, beta_min=0.00001, beta_max=1.0):
        """
        Generates a list of n beta values, each updated linearly.

        Parameters:
        - beta_start (float): Initial beta value.
        - beta_end (float): Final beta value.
        - n (int): Number of elements to generate.
        - beta_min (float): Minimum beta value.
        - beta_max (float): Maximum beta value.

        Returns:
        - beta_schedule (list): List of n beta values.
        """
        beta_schedule = []
        step = (beta_end - beta_start) / n # Calculate the increment for each step
        beta = beta_start
        for _ in range(n):
        beta = min(beta_max, max(beta_min, beta))
        beta_schedule.append(beta)
        beta += step
        return beta_schedule
    '''
    
    def update(self, idxs, td_errors):
        td_errors = torch.abs(td_errors).view(-1)
        for idx, td_error in zip(idxs, td_errors):
            self.memory[idx][self.td_error_index] = td_error.item()
        if self.rank_based:
            self.memory[: self.n_entries] = sorted(
                self.memory[: self.n_entries],
                key=lambda x: x[self.td_error_index],
                reverse=True,
            )

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priorities = [
                entry[self.td_error_index] for entry in self.memory[: self.n_entries]
            ]
            priority = max(priorities)
        entry = [priority, sample]
        if self.n_entries < self.max_samples:
            self.memory.append(entry)
        else:
            self.memory[self.next_index] = entry
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index = (self.next_index + 1) % self.max_samples

    
    def _update_beta(self):
        
        #self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        
        try:
            self.beta = self.betas[self.beta_index]
        except:
            self.beta = self.beta_end
        self.beta_index += 1
        
        
        return self.beta
    
    '''
    def _update_beta(self):
        if self.current_step < self.beta_steps:
            self.beta *= self.beta_rate
            self.beta = min(self.beta, self.beta_end)
            self.current_step += 1
        else:
            self.beta = self.beta_end
        return self.beta
    '''

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self._update_beta()
        
        entries = self.memory[: self.n_entries]

        if self.rank_based:
            priorities = 1.0 / (torch.arange(self.n_entries) + 1).float()
        else:
            # Based on TD error priority allocation
            priorities = torch.tensor(
                [entry[self.td_error_index] for entry in entries], dtype=torch.float32, device = self.device
            )
            priorities += self.EPS

        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / torch.sum(scaled_priorities)

        weights = (self.n_entries * probs) ** -self.beta
        normalized_weights = weights / torch.max(weights)

        idxs = torch.multinomial(probs, batch_size, replacement=False)
        samples = [entries[idx][self.sample_index] for idx in idxs]

        idxs_stack = idxs.unsqueeze(1)
        weights_stack = normalized_weights[idxs].unsqueeze(1)

        return idxs_stack, weights_stack, samples

    def __len__(self):
        return self.n_entries

class PER:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = PrioritizedReplayBuffer(
            max_samples=capacity, alpha=alpha, beta0=beta
        )
        self.beta = beta
        self.beta_increment = beta_increment

    def add(self, priority, experience):
        self.buffer.store(experience)

    def sample(self, batch_size):
        samples = self.buffer.sample(batch_size=batch_size)
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples

    def update(self, idx, priority):
        self.buffer.update(idx, priority)