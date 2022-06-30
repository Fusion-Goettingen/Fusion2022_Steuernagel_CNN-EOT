from abc import ABC, abstractmethod


class AbstractTracker(ABC):
    """Abstract base class that should be implemented by Trackers"""
    @abstractmethod
    def predict(self):
        """Predict forward in time"""
        pass

    @abstractmethod
    def update(self, Z):
        """Update with N Measurements (parameter Z), given as list of shape (N,2)"""
        pass

    @abstractmethod
    def get_state(self):
        """Return the current state in 7D: [loc_x, loc_y, velo_x, velo_y, orientation, length, width]"""
        pass

    @abstractmethod
    def set_R(self, R):
        """Update the measurement noise covariance matrix R"""
        pass
