import numpy as np
import random
from scipy.stats import norm, multivariate_normal
from collections import defaultdict
# Get the project root directory
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)
from Datasets.wsdream.Read_Dataset import probability_sampling

class UncontrollableVariableSampler:
    def __init__(self):
        pass

    
    def sample_from_discrete_distribution(self, distribution):
        """
        Samples a value from a given discrete probability distribution.
        
        :param distribution: A dictionary representing a discrete distribution 
                             where keys are values and values are probabilities.
        :return: A sampled value from the distribution.
        """
                
        total_prob = sum(distribution.values())
        if not np.isclose(total_prob, 1.0):
            raise ValueError("The sum of probabilities must be 1.")
        return np.random.choice(list(distribution.keys()), p=list(distribution.values()))

    # here we use Gaussian/Normal distribution as an example
    def sample_from_continuous_distribution(self, mean, std_dev):
        """
        Samples a value from a continuous probability distribution (Gaussian/Normal).
        
        :param mean: Mean of the distribution.
        :param std_dev: Standard deviation of the distribution.
        :return: A sampled value from the continuous distribution.
        """
        return np.random.normal(loc=mean, scale=std_dev)

    def sample_from_joint_distribution(self, joint_distribution):
        """
        Samples a value from a given joint probability distribution.
        
        :param joint_distribution: A dictionary representing a joint distribution 
                                   where keys are tuples of values and values are probabilities.
        :return: A sampled tuple of values from the joint distribution.
        """
        return np.random.choice(list(joint_distribution.keys()), p=list(joint_distribution.values()))

    def get_value(self, var, current_state):
        """
        Retrieves the value of an uncontrollable variable based on its probability distribution.
        
        :param var: The uncontrollable variable whose value needs to be determined.
        :param current_state: The current state of the PMDP, can be used for conditioning.
        :return: The sampled value(s) of the uncontrollable variable.
        """
        
        if isinstance(var, dict) and 'joint_distribution' in var:
            # Case 3: var is a joint probability distribution involving multiple variables
            joint_distribution = var['joint_distribution']
            sampled_values = self.sample_from_joint_distribution(joint_distribution)
            return sampled_values
        
        elif isinstance(var, dict) and 'discrete_distribution' in var:
            # Case 1: var is an independent discrete probability distribution
            discrete_distribution = var['discrete_distribution']
            sampled_value = self.sample_from_discrete_distribution(discrete_distribution)
            return sampled_value
        
        elif isinstance(var, dict) and 'continuous_distribution' in var:
            # Case 2: var is an independent continuous probability distribution
            mean = var['continuous_distribution']['mean']
            std_dev = var['continuous_distribution']['std_dev']
            sampled_value = self.sample_from_continuous_distribution(mean, std_dev)
            return sampled_value

        else:
            raise ValueError("Unknown variable format or distribution type for var")
        
        
    def define_joint_distribution(self, variables, correlations):
        """
        Define a multivariate normal distribution for continuous variables or 
        a custom joint distribution for discrete variables.
        
        :param variables: A list of variables to be included in the joint distribution.
                          Each variable is represented by a tuple (mean, std) for continuous 
                          or (values, probabilities) for discrete.
        :param correlations: Correlation matrix for continuous variables, or None for discrete variables.
        :return: A dictionary representing the joint distribution.
        """
        if correlations is None:
            # Discrete joint distribution
            joint_distribution = {}
            for values, prob in zip(*variables):
                joint_distribution[tuple(values)] = prob
            return {'joint_distribution': joint_distribution}
        
        else:
            # Continuous joint distribution (Multivariate Normal)
            means, std_devs = zip(*variables)
            covariance_matrix = np.outer(std_devs, std_devs) * correlations
            mvn = multivariate_normal(mean=means, cov=covariance_matrix)
            return mvn

    # until, 01/09/2024, we only use the following function 'sample_based_on_datset_and_current_state'
    # Each call to sample_based_on_current_state() will sample a value for a single variable (i.e., variable_to_sample) based on the current state
    def sample_based_on_datset_and_current_state(self, dataset, var_name_to_sample, current_state):
        # Extract the value of a known variable
        known_values = {var: val for var, val in current_state.items() if val is not None}

        # Filter the dataset to keep samples that match the known variable values
        # Here we assume dataset is a list, and each element is a dictionary representing a sample
        # known_values is subset of variables, compatible with a sample
        # each sample must contain all variables in known_values and variable_to_sample
        filtered_dataset = [
            sample for sample in dataset
            if all(sample.get(var) == val for var, val in known_values.items()) and (var_name_to_sample in sample)
        ]

        # Calculate the conditional distribution
        # We based on the known values about variables and the dataset to calculate the conditional distribution
        # defaultdict(int) will set the default value of a key to 0 when it is accessed for the first time and does not exist
        conditional_distribution = defaultdict(int)
        for sample in filtered_dataset:
            #key = tuple(sample[var] for var in var_name_to_sample)
            key = sample[var_name_to_sample]
            conditional_distribution[key] += 1
        
        total_count = sum(conditional_distribution.values())
        for key in conditional_distribution:
            conditional_distribution[key] /= total_count
        
        # Sampling
        # random.choices() is used to sample a value from a list of values with given probabilities
        # population: Specify the sequence to choose from (can be a list, tuple, etc.)
        # weights: Specify the weight (probability) for each element. If not specified, the weight is equal by default,
        # and the length of weights must be equal to the length of population, but not necessary to sum to 1
        # the variable_to_sample contains one variable, so we only need to sample one value
        sampled_value = random.choices(
            population=list(conditional_distribution.keys()),
            weights=list(conditional_distribution.values())
        )[0]
        
        return dict(zip(var_name_to_sample, sampled_value))
    
    def sample_QoS_CSSC_wsdream_dataset(probabilities, unique_values):

        
        sampled_value = probability_sampling(probabilities, unique_values, sample_size=1, replace=True)[0]
        return sampled_value

if __name__ == "__main__":
    sampler = UncontrollableVariableSampler()

    # Example 1: Independent discrete distributions
    var_dist = {
        'discrete_distribution': {'low': 0.4, 'medium': 0.4, 'high': 0.2}
    }
    sampled_value = sampler.get_value(var_dist, current_state=None)
    print("Sampled Value (Independent Discrete):", sampled_value)

    # Example 2: Independent continuous distributions
    var_continuous = {
        'continuous_distribution': {'mean': 0, 'std_dev': 1}
    }
    sampled_continuous_value = sampler.get_value(var_continuous, current_state=None)
    print("Sampled Value (Independent Continuous):", sampled_continuous_value)


    # Example 3: Continuous Joint Distribution
    variables = [(0, 1), (3, 2)]  # (mean, std_dev) for each variable
    correlations = [[1, 0.5], [0.5, 1]]  # Correlation matrix
    mvn_dist = sampler.define_joint_distribution(variables, correlations)
    sampled_continuous_values = mvn_dist.rvs(size=1)
    print("Sampled Continuous Values (Multivariate Normal):", sampled_continuous_values)
