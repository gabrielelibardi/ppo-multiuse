import time
import random
from arenas import cl_tasks


class CLManager:
    """
    A class to manage which arenas are generated for curriculum learning.
    """

    def __init__(self, threshold=0.75):
        """
        Initialize a Manager.
        """

        self.current_level = 1
        self.threshold = threshold
        self.generators = cl_tasks["generators"]
        self.num_levels = len(cl_tasks["generators"])
        self.generator_parameters = cl_tasks["parameters"]
        self.generator_pool = [self.generators[self.current_level]]
        self.params_pool = [self.generator_parameters[self.current_level]]

    def sample_arena_from_current_pool(self):
        """ Return an arena generator function from the current pool. """

        index = random.choice(range(len(self.generator_pool)))
        return self.generator_pool[index], self.params_pool[index], index

    def update_pool(self, performance):
        """ If reached goal performance, add a harder task to the pool. """

        if performance > self.threshold \
                and self.current_level < self.num_levels:
            self.current_level += 1
            self.generator_pool.append(self.generators[self.current_level])
            self.params_pool.append(self.generator_parameters[self.current_level])

        return self.current_level


if __name__ == "__main__":

    # Create Manager
    manager = CLManager()
