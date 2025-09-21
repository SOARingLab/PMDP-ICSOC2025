import numpy as np
import heapq, numbers
import cupy as cp
import torch


class PrioritizedReplayBuffer():
    def __init__(self, 
                 max_samples=10000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992):
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=object)
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.EPS = 1e-6
        self.explore_prob = 0.1


    def update(self, idxs, td_errors):
        # Make sure the shape of td_errors matches self.memory[idxs, self.td_error_index]
        td_errors = np.abs(td_errors).reshape(-1, 1)
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            # If using ranking sampling, sort by TD error
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def update_cupy(self, idxs, td_errors):
        # Transferring TD Error to GPU
        td_errors_gpu = cp.asarray(td_errors)
        
        # Calculate the absolute value and update memory
        abs_td_errors_gpu = cp.abs(td_errors_gpu).reshape(-1, 1)
        self.memory[idxs, self.td_error_index] = cp.asnumpy(abs_td_errors_gpu)  # update memory

        if self.rank_based:
            # Sorting on the GPU
            sorted_indices = cp.argsort(abs_td_errors_gpu)[::-1]  # From big to small
            # Update the contents of memory
            self.memory[:self.n_entries] = self.memory[idxs[cp.asnumpy(sorted_indices)][:self.n_entries]]

    def store(self, sample):
        # Each time a new experience is stored [state, action, reward, next_state, done]
        sample = np.array(sample, dtype=object)
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority
        self.memory[self.next_index, 
                    self.sample_index] = sample
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, epsilons, e, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries) + 1)
        else:
            priorities = entries[:, self.td_error_index] + self.EPS
            #priorities = priorities.astype(np.float64)
            #priorities = np.where(np.isnan(priorities) | np.isinf(priorities), self.EPS, priorities)


        scaled_priorities = priorities**self.alpha        
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)

        
        #if np.random.rand() < epsilons[e]:
            # Perform random sampling
            #idxs = np.random.choice(self.n_entries, batch_size, replace=False)
        #else:
            # Execution priority sampling
            #idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)

        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        
        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])
    


class PER:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = PrioritizedReplayBuffer(capacity, alpha)
        self.beta = beta
        self.beta_increment = beta_increment

    def add(self, priority, experience):
        self.buffer.add(priority, experience)

    def sample(self, batch_size):
        beta = min(1.0, self.beta)
        samples = self.buffer.sample(batch_size, beta)
        self.beta += self.beta_increment
        return samples

    def update(self, idx, priority):
        self.buffer.update(idx, priority)