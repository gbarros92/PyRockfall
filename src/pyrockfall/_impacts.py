"""
Impacts data structure for pyrockfall
======================================

Records the time history of block impacts against slope elements during
rockfall simulations: impact location, orientation, pre- and post-impact
velocities, and restitution coefficients.
"""

import numpy as np

class Impacts:
    """
    Records and processes the time history of block impacts during
    rockfall simulations.
    """

    def __init__(self):
        """
        Initializes an empty Impacts instance for storing impact history.
        """
        self._time_history = []
        self._position_history = []
        self._normal_history = []
        self._element_id_history = []
        self._normal_impact_velocity_history = []
        self._tangential_impact_velocity_history = []
        self._angular_impact_velocity_history = []
        self._normal_post_velocity_history = []
        self._tangential_post_velocity_history = []
        self._angular_post_velocity_history = []
        self._normal_restitution_history = []
        self._tangential_restitution_history = []


    def __del__(self):
        """
        Resets all internal history to their default values.
        """
        self._time_history = []
        self._position_history = []
        self._normal_history = []
        self._element_id_history = []
        self._normal_impact_velocity_history = []
        self._tangential_impact_velocity_history = []
        self._angular_impact_velocity_history = []
        self._normal_post_velocity_history = []
        self._tangential_post_velocity_history = []
        self._angular_post_velocity_history = []
        self._normal_restitution_history = []
        self._tangential_restitution_history = []

    def addData(self, time, position, normal, element_id,
                normal_impact_velocity, tangential_impact_velocity, angular_impact_velocity,
                normal_post_velocity, tangential_post_velocity, angular_post_velocity,
                normal_restitution, tangential_restitution):
        """
        Adds a new impact event to the internal history.

        Args:
            time (array-like): Timestamps of each impact.
            position (array-like): Impact positions (dim x blocks).
            normal (array-like): Impact surface normals (dim x blocks).
            element_id (array-like): Slope element ID of each impact.
            normal_impact_velocity (array-like): Normal velocity before impact.
            tangential_impact_velocity (array-like): Tangential velocity before impact.
            angular_impact_velocity (array-like): Angular velocity before impact.
            normal_post_velocity (array-like): Normal velocity after impact.
            tangential_post_velocity (array-like): Tangential velocity after impact.
            angular_post_velocity (array-like): Angular velocity after impact.
            normal_restitution (array-like): Normal restitution coefficient.
            tangential_restitution (array-like): Tangential restitution coefficient.
        """
        # Ensure histories are lists before appending
        if not isinstance(self._time_history, list):
            self._time_history = list(self._time_history)
        if not isinstance(self._position_history, list):
            self._position_history = list(self._position_history)
        if not isinstance(self._normal_history, list):
            self._normal_history = list(self._normal_history)
        if not isinstance(self._element_id_history, list):
            self._element_id_history = list(self._element_id_history)
        if not isinstance(self._normal_impact_velocity_history, list):
            self._normal_impact_velocity_history = list(self._normal_impact_velocity_history)
        if not isinstance(self._tangential_impact_velocity_history, list):
            self._tangential_impact_velocity_history = list(self._tangential_impact_velocity_history)
        if not isinstance(self._angular_impact_velocity_history, list):
            self._angular_impact_velocity_history = list(self._angular_impact_velocity_history)
        if not isinstance(self._normal_post_velocity_history, list):
            self._normal_post_velocity_history = list(self._normal_post_velocity_history)
        if not isinstance(self._tangential_post_velocity_history, list):
            self._tangential_post_velocity_history = list(self._tangential_post_velocity_history)
        if not isinstance(self._angular_post_velocity_history, list):
            self._angular_post_velocity_history = list(self._angular_post_velocity_history)
        if not isinstance(self._normal_restitution_history, list):
            self._normal_restitution_history = list(self._normal_restitution_history)
        if not isinstance(self._tangential_restitution_history, list):
            self._tangential_restitution_history = list(self._tangential_restitution_history)

        self._time_history.append(np.array(time))
        self._position_history.append(np.array(position))
        self._normal_history.append(np.array(normal))
        self._element_id_history.append(np.array(element_id))
        self._normal_impact_velocity_history.append(np.array(normal_impact_velocity))
        self._tangential_impact_velocity_history.append(np.array(tangential_impact_velocity))
        self._angular_impact_velocity_history.append(np.array(angular_impact_velocity))
        self._normal_post_velocity_history.append(np.array(normal_post_velocity))
        self._tangential_post_velocity_history.append(np.array(tangential_post_velocity))
        self._angular_post_velocity_history.append(np.array(angular_post_velocity))
        self._normal_restitution_history.append(np.array(normal_restitution))
        self._tangential_restitution_history.append(np.array(tangential_restitution))


    def simulationDone(self):
        """
        Finalizes the data collection by converting all internal lists to NumPy arrays.
        """
        self._time_history = np.asarray(self._time_history)
        self._position_history = np.asarray(self._position_history)
        self._normal_history = np.asarray(self._normal_history)
        self._element_id_history = np.asarray(self._element_id_history)
        self._normal_impact_velocity_history = np.asarray(self._normal_impact_velocity_history)
        self._tangential_impact_velocity_history = np.asarray(self._tangential_impact_velocity_history)
        self._angular_impact_velocity_history = np.asarray(self._angular_impact_velocity_history)
        self._normal_post_velocity_history = np.asarray(self._normal_post_velocity_history)
        self._tangential_post_velocity_history = np.asarray(self._tangential_post_velocity_history)
        self._angular_post_velocity_history = np.asarray(self._angular_post_velocity_history)
        self._normal_restitution_history = np.asarray(self._normal_restitution_history)
        self._tangential_restitution_history = np.asarray(self._tangential_restitution_history)

    @property
    def timeHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Time history of impacts.
        """
        return np.asarray(self._time_history)

    @property
    def positionHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Position history of impacts.
        """
        return np.asarray(self._position_history)

    @property
    def normalHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Surface normal history of impacts.
        """
        return np.asarray(self._normal_history)

    @property
    def elementIdHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Slope element ID history of impacts.
        """
        return np.asarray(self._element_id_history)

    @property
    def normalImpactVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Normal velocity history before impact.
        """
        return np.asarray(self._normal_impact_velocity_history)

    @property
    def tangentialImpactVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Tangential velocity history before impact.
        """
        return np.asarray(self._tangential_impact_velocity_history)

    @property
    def angularImpactVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Angular velocity history before impact.
        """
        return np.asarray(self._angular_impact_velocity_history)

    @property
    def normalPostVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Normal velocity history after impact.
        """
        return np.asarray(self._normal_post_velocity_history)

    @property
    def tangentialPostVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Tangential velocity history after impact.
        """
        return np.asarray(self._tangential_post_velocity_history)

    @property
    def angularPostVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Angular velocity history after impact.
        """
        return np.asarray(self._angular_post_velocity_history)

    @property
    def normalRestitutionHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Normal restitution coefficient history.
        """
        return np.asarray(self._normal_restitution_history)

    @property
    def tangentialRestitutionHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Tangential restitution coefficient history.
        """
        return np.asarray(self._tangential_restitution_history)

    @property
    def numberOfBlocks(self) -> int:
        """
        Returns:
            int: Number of blocks simulated.
        """
        return self.timeHistory.shape[1]

    @property
    def numberOfImpacts(self) -> int:
        """
        Returns:
            int: Number of impact events recorded.
        """
        return self.timeHistory.shape[0]

    @property
    def impactPoints(self):
        """
        Returns:
            np.ndarray: Position history of all blocks in shape (dim, impact #, blocks).
        """
        return np.transpose(np.array(self.positionHistory), (1, 0, 2))

    @property
    def impactNormalVelocities(self):
        """
        Returns:
            np.ndarray: Normal impact velocity history, transposed to shape (impact #, blocks).
        """
        return np.transpose(np.array(self.normalImpactVelocityHistory))

    @property
    def impactTangentialVelocities(self):
        """
        Returns:
            np.ndarray: Tangential impact velocity history, transposed to shape (impact #, blocks).
        """
        return np.transpose(np.array(self.tangentialImpactVelocityHistory))

    @property
    def impactAngularVelocities(self):
        """
        Returns:
            np.ndarray: Angular impact velocity history in shape (dim, impact #, blocks)
            (dim is 1 for 2D angular velocities).
        """
        return np.transpose(np.array(self.angularImpactVelocityHistory), (1, 0, 2))
