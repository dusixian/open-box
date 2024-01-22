# used for early stop

class EarlyStopException(Exception):
    """Exception raised for early stopping in Advisor."""
    pass

class EarlyStopAlgorithm:
    def __init__(self, early_stop, config_space, early_stop_kwargs):
        self.early_stop = early_stop
        self.config_space = config_space
        if(early_stop_kwargs):
            self.early_stop_kwargs = early_stop_kwargs
        else:
            self.early_stop_kwargs = dict({})
        self.last_eta = None
        self.no_improvement_rounds = 0
        self.default_obj_value = None
        self.already_early_stopped = False


    def check_setup(self, advisor):
        """
        Check if the early stopping algorithm is applicable to the given advisor.
        """
        if advisor.num_objectives != 1:
            raise ValueError("Early stopping is only supported for single-objective optimization.")

        if('improvement_threshold' in self.early_stop_kwargs):
            if self.early_stop_kwargs['improvement_threshold'] <= 0:
                raise ValueError("Improvement Threshold early stopping requires threshold to be larger than 0.")
            if advisor.acq_type != 'ei':
                raise ValueError("Improvement Threshold early stopping requires the Expected Improvement acquisition function.")
            
        if(not ('improvement_threshold' in self.early_stop_kwargs) and not ('no_improvement_rounds' in self.early_stop_kwargs)):
            raise ValueError("Early stopping requires at least one of the following arguments: improvement_threshold, no_improvement_rounds.")
        
        if('min_iter' in self.early_stop_kwargs and self.early_stop_kwargs['min_iter'] < 0):
            raise ValueError("Minimum number of iterations for early stopping must be non-negative.")


    def should_early_stop(self, history, config, acquisition_function):
        """
        Determine whether to early stop based on the given criteria.
        """
        if not self.early_stop:
            return False
        
        if self.already_early_stopped:
            return True

        # check if reach the minimum number of iter
        min_iter = self.early_stop_kwargs.get('min_iter', 10)
        if len(history) < min_iter:
            return False

        # Condition 1: EI less than 10% of the difference between the best and default configuration
        if self.early_stop_kwargs.get('improvement_threshold', 0) > 0:
            acq = acquisition_function([config], convert=True)[0]
            best_obs = acquisition_function.eta
            if self.default_obj_value is None:
                default_config = self.config_space.get_default_configuration()
                for obs in history.observations:
                    if obs.config == default_config:
                        self.default_obj_value = obs.objectives
                        break
                assert self.default_obj_value is not None

            improvement = self.default_obj_value[0] - best_obs
            if acq[0] < self.early_stop_kwargs['improvement_threshold'] * improvement:
                return True

        # Condition 2: No improvement over multiple rounds
        if 'no_improvement_rounds' in self.early_stop_kwargs:
            if self.last_eta is None or best_obs < self.last_eta:
                self.last_eta = best_obs
                self.no_improvement_rounds = 0
            else:
                self.no_improvement_rounds += 1
                if self.no_improvement_rounds >= self.early_stop_kwargs['no_improvement_rounds']:
                    return True

        return False
