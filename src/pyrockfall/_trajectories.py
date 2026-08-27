"""
Trajectories data structure for pyrockfall
==========================================

Records and processes the time history of block positions, velocities,
accelerations, and energy during rockfall simulations.

Provides tools for analysis, file export/import, and visualization of
simulation results.
"""

import numpy as np
from numbers import Real
import gzip
import os

class Trajectories:
    """
    Records and processes the time history of block positions, velocities,
    accelerations, and energy during rockfall simulations.

    Provides utilities for data analysis, exporting, and visualization.
    """

    def __init__(self):
        """
        Initializes an empty Trajectories instance for storing simulation history.
        Sets default values for mass, inertia, floor height, and gravity.
        """
        self._time_history = []
        self._position_history = []
        self._velocity_history = []
        self._angular_velocity_history = []
        self._acceleration_history = []
        self.mass = 0.0
        self.inertia = 0.0
        self.floor = 0.0
        self.gravity = 9.80665


    def __del__(self):
        """
        Resets all internal history and physical parameters to their default values.
        """
        self._time_history = []
        self._position_history = []
        self._velocity_history = []
        self._angular_velocity_history = []
        self._acceleration_history = []
        self.mass = 0.0
        self.inertia = 0.0
        self.floor = 0.0
        self.gravity = 9.80665

    def addData(self, time, positions, velocities, angular_velocities, accelerations):
        """
        Adds a new time step of simulation data to the internal history.

        Args:
            time (array-like): Timestamps for each block.
            positions (array-like): Block positions (dim x blocks).
            velocities (array-like): Block velocities (dim x blocks).
            angular_velocities (array-like): Angular velocities for each block.
            accelerations (array-like): Block accelerations (dim x blocks).
        """
        # Ensure histories are lists before appending
        if not isinstance(self._time_history, list):
            self._time_history = list(self._time_history)
        if not isinstance(self._position_history, list):
            self._position_history = list(self._position_history)
        if not isinstance(self._velocity_history, list):
            self._velocity_history = list(self._velocity_history)
        if not isinstance(self._angular_velocity_history, list):
            self._angular_velocity_history = list(self._angular_velocity_history)
        if not isinstance(self._acceleration_history, list):
            self._acceleration_history = list(self._acceleration_history)

        self._time_history.append(np.array(time))
        self._position_history.append(np.array(positions))
        self._velocity_history.append(np.array(velocities))
        self._angular_velocity_history.append(np.array(angular_velocities))
        self._acceleration_history.append(np.array(accelerations))


    def simulationDone(self):
        """
        Finalizes the data collection by converting all internal lists to NumPy arrays.
        """
        self._time_history = np.asarray(self._time_history)
        self._position_history = np.asarray(self._position_history)
        self._velocity_history = np.asarray(self._velocity_history)
        self._angular_velocity_history = np.asarray(self._angular_velocity_history)
        self._acceleration_history = np.asarray(self._acceleration_history)
    
    @property
    def timeHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Time history of the simulation.
        """
        return np.asarray(self._time_history)

    @property
    def positionHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Position history of the simulation.
        """
        return np.asarray(self._position_history)

    @property
    def velocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Velocity history of the simulation.
        """
        return np.asarray(self._velocity_history)

    @property
    def angularVelocityHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Angular velocity history of the simulation.
        """
        return np.asarray(self._angular_velocity_history)

    @property
    def accelerationHistory(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Acceleration history of the simulation.
        """
        return np.asarray(self._acceleration_history)

    @property
    def numberOfBlocks(self) -> int:
        """
        Returns:
            int: Number of blocks simulated.
        """
        return self.timeHistory.shape[1]


    @property
    def impactPoints(self):
        """
        Returns:
            np.ndarray: Position history of all blocks in shape (dim, impact #, blocks).
        """
        return np.transpose(np.array(self.positionHistory), (1, 0, 2))


    @property
    def impactVelocities(self):
        """
        Returns:
            np.ndarray: Velocity history of all blocks in shape (dim, impact #, blocks).
        """
        return np.transpose(np.array(self.velocityHistory), (1, 0, 2))


    @property
    def impactAngularVelocities(self):
        """
        Returns:
            np.ndarray: Angular velocity history of shape (dim, impact #, blocks).
        """
        return np.transpose(np.array(self.angularVelocityHistory), (1, 0, 2))


    @property
    def endpoints(self):
        """
        Returns:
            np.ndarray: Final positions of all blocks.
        """
        return np.array(self.positionHistory[-1, :, :])


    @property
    def startTime(self):
        """
        Returns:
            np.ndarray: Start time for each block.
        """
        return self.timeHistory[0]


    @property
    def stopTime(self):
        """
        Returns:
            np.ndarray: Stop time for each block.
        """
        return self.timeHistory[-1]


    def position(self, time):
        """
        Returns the position of the blocks at the specified time(s).

        Args:
            time (float or np.ndarray): Time(s) at which to compute the position.

        Returns:
            np.ndarray: Interpolated positions (dim x time x blocks).
        """
        time = np.asarray(time)
        if time.ndim == 0:  # Single float value
            time = np.full((1, self.positionHistory.shape[2]), time)
        elif time.ndim == 1:  # 1D array
            time = np.tile(time[:, np.newaxis], (1, self.positionHistory.shape[2]))        
        num_requested_times, num_samples = time.shape
        space_dim = self.positionHistory.shape[1]
        pos = np.zeros((space_dim, num_requested_times, num_samples))        
        for sample in range(num_samples):
            sample_time = time[:, sample]
            time_history_sample = self.timeHistory[:, sample]
            
            # Find the indices of the time instants just before the desired times
            indices = np.searchsorted(time_history_sample, sample_time, side='right')
            indices = np.clip(indices, 1, len(time_history_sample) - 1)

            # Calculate the time difference between instants
            Dt = time_history_sample[indices] - time_history_sample[indices-1]
            
            # Get the positions, velocities, and accelerations at the found indices
            pos_sample = np.transpose(self.positionHistory[indices-1, :, sample])
            vel_sample = np.transpose(self.velocityHistory[indices, :, sample])
            acc_sample = np.transpose(self.accelerationHistory[indices, :, sample])

            vel_sample -= acc_sample * Dt
            
            # Calculate the time differences
            dt = sample_time - time_history_sample[indices-1]
            dt = np.where(dt < Dt, dt, Dt)
            
            # Calculate the new positions
            pos[:, :, sample] = pos_sample + vel_sample * dt + 0.5 * acc_sample * dt**2

        pos = np.where(np.isnan(time), time, pos)
        
        if num_requested_times == 1:
            pos = pos[:, 0, :]
        return pos


    def velocity(self, time, angVel=False):
        """
        Returns the velocity of the blocks at the specified time(s).

        Args:
            time (float or np.ndarray): Time(s) at which to compute the velocity.
            angVel (bool): If True, also return angular velocity.

        Returns:
            np.ndarray or tuple: Velocities or (velocities, angular velocities).
        """
        time = np.asarray(time)
        if time.ndim == 0:  # Single float value
            time = np.full((1, self.positionHistory.shape[2]), time)
        elif time.ndim == 1:  # 1D array
            time = np.tile(time[:, np.newaxis], (1, self.positionHistory.shape[2]))
        num_requested_times, num_samples = time.shape
        space_dim = self.positionHistory.shape[1]
        vel = np.zeros((space_dim, num_requested_times, num_samples))
        w  = np.zeros((space_dim if space_dim == 3 else 1, num_requested_times, num_samples))
        for sample in range(num_samples):
            sample_time = time[:, sample]
            time_history_sample = self.timeHistory[:, sample]
            
            # Find the indices of the time instants just before the desired times
            indices = np.searchsorted(time_history_sample, sample_time, side='right')
            indices = np.clip(indices, 1, len(time_history_sample) - 1)

            # Calculate the time difference between instants
            Dt = time_history_sample[indices] - time_history_sample[indices-1]
            
            # Get the positions, velocities, and accelerations at the found indices
            vel_sample = np.transpose(self.velocityHistory[indices, :, sample])
            acc_sample = np.transpose(self.accelerationHistory[indices, :, sample])

            vel_sample -= acc_sample * Dt
            
            # Calculate the time differences
            dt = sample_time - time_history_sample[indices-1]
            dt = np.where(dt < Dt, dt, Dt)
            
            # Calculate the new velocities
            vel[:, :, sample] = vel_sample + acc_sample * dt
            w[:, :, sample]  = np.transpose(self.angularVelocityHistory[indices, :, sample])

        vel = np.where(np.isnan(time), time, vel)
        w = np.where(np.isnan(time), time, w)
        
        if num_requested_times == 1:
            vel = vel[:, 0, :]
            w  = w[0, :]
        if angVel:
            return vel, w
        else:
            return vel
        
    
    def acceleration(self, time):
        """
        Returns the acceleration of the blocks at the specified time(s).

        Args:
            time (float or np.ndarray): Time(s) at which to compute the acceleration.

        Returns:
            np.ndarray: Acceleration values (dim x time x blocks).
        """
        time = np.asarray(time)
        if time.ndim == 0:  # Single float value
            time = np.full((1, self.positionHistory.shape[2]), time)
        elif time.ndim == 1:  # 1D array
            time = np.tile(time[:, np.newaxis], (1, self.positionHistory.shape[2]))
        num_requested_times, num_samples = time.shape
        space_dim = self.positionHistory.shape[1]
        acc = np.zeros((space_dim, num_requested_times, num_samples))
        for sample in range(num_samples):
            sample_time = time[:, sample]
            time_history_sample = self.timeHistory[:, sample]
            
            # Find the indices of the time instants just before the desired times
            indices = np.searchsorted(time_history_sample, sample_time, side='right')
            indices = np.clip(indices, 1, len(time_history_sample) - 1)
            
            # Get the positions, velocities, and accelerations at the found indices
            acc_sample = np.transpose(self.accelerationHistory[indices, :, sample])
            
            # Calculate the new velocities
            acc[:, :, sample] = acc_sample

        acc = np.where(np.isnan(time), time, acc)
        
        if num_requested_times == 1:
            acc = acc[:, 0, :]
        return acc
    
    
    def collector(self, points, normals):
        """
        Compute the first intersection time between each block's trajectory and
        each collector plane/line (works for 2D and 3D trajectories alike).

        Parameters:
            points (np.ndarray): Points on each collector (num_dim, num_collectors)
            normals (np.ndarray): Normal vectors defining collector orientation (num_dim, num_collectors)

        Returns:
            np.ndarray: First crossing times (collectors x blocks), NaN if no crossing.
        """
        points = np.asarray(points)
        normals = np.asarray(normals)
        if points.ndim == 1:
            points = points[:, None]
            normals = normals[:, None]
        T, D, P = self.positionHistory.shape
        C = points.shape[1]

        # Expand dims for broadcasting
        # (T, D, P, 1) - (1, D, 1, C) -> (T, D, P, C)
        positions = self.positionHistory[:, :, :, None]
        points = points[None, :, None, :]
        normals = normals[None, :, None, :]

        # Compute projection of vector from collector point to particle position onto normal
        rel_pos = positions - points  # (T, D, P, C)
        projections = np.einsum('tdpc,tdpc->tpc', rel_pos, normals)  # (T, P, C)

        # Detect sign changes over time (crossings)
        sign_changes = np.diff(np.sign(projections), axis=0)
        t_idx, p_idx, c_idx = np.where(sign_changes != 0)

        crossing_times = np.full((C, P), np.nan)
        if t_idx.size == 0:
            return crossing_times

        # Prepare segment data for interpolation from t_idx to t_idx+1
        t1 = t_idx + 1

        # Dt = t1 - t0
        t0 = self.timeHistory[t_idx, p_idx]
        t1_val = self.timeHistory[t1, p_idx]
        Dt = t1_val - t0

        # Get final position p1 (stored), and time t1
        vf = self.velocityHistory[t1, :, p_idx].T     # (D, N)
        a = self.accelerationHistory[t1, :, p_idx].T   # (D, N)
        v0 = vf - a * Dt                                  # Recovered initial velocity
        p0 = self.positionHistory[t_idx, :, p_idx].T   # (D, N)

        pts_cross = points[0, :, 0, c_idx].T              # (D, N)
        nrm_cross = normals[0, :, 0, c_idx].T             # (D, N)

        dp = p0 - pts_cross

        A = 0.5 * np.einsum('ij,ij->j', a, nrm_cross)
        B = np.einsum('ij,ij->j', v0, nrm_cross)
        Cc = np.einsum('ij,ij->j', dp, nrm_cross)

        A_is_zero = np.isclose(A, 0.0)
        dt = np.full_like(A, np.nan)

        # Linear case: A == 0 → dt = -C / B
        with np.errstate(divide='ignore', invalid='ignore'):
            dt_linear = -Cc[A_is_zero] / B[A_is_zero]
        dt[A_is_zero] = np.where(dt_linear > 0, dt_linear, np.nan)

        # Quadratic case: A ≠ 0
        if np.any(~A_is_zero):
            A_q = A[~A_is_zero]
            B_q = B[~A_is_zero]
            C_q = Cc[~A_is_zero]

            discriminant = B_q**2 - 4*A_q*C_q
            discriminant[discriminant < 0] = 0.0
            sqrt_disc = np.sqrt(discriminant)

            with np.errstate(divide='ignore', invalid='ignore'):
                dt1 = (-B_q - sqrt_disc) / (2*A_q)
                dt2 = (-B_q + sqrt_disc) / (2*A_q)

            dt_quad = np.vstack([dt1, dt2])
            dt_quad[dt_quad < 0] = np.nan
            dt[~A_is_zero] = np.nanmin(dt_quad, axis=0)

        valid = ~np.isnan(dt)
        t_cross = t0[valid] + dt[valid]

        # A block can cross a collector's plane more than once (e.g. it
        # bounces back across it later). `t_idx`, and therefore every array
        # derived from it above (including c_idx[valid]/p_idx[valid]/t_cross
        # here), is already ordered by increasing segment index because
        # np.where scans axis 0 (time) outermost. Segments are disjoint,
        # non-overlapping time intervals, so this is equivalently an
        # ascending-time order. Assigning in reverse means that, for any
        # (collector, block) pair hit more than once, the later crossings
        # are written first and the earliest crossing is written last -
        # and therefore wins, without needing an explicit sort.
        # NB: the reversal must be materialised with .copy() - assigning
        # through a negative-stride *view* does not reliably preserve the
        # write order for duplicate fancy indices.
        c_rev = c_idx[valid][::-1].copy()
        p_rev = p_idx[valid][::-1].copy()
        t_rev = t_cross[::-1].copy()
        crossing_times[c_rev, p_rev] = t_rev

        return crossing_times

    
    def plot(self, ax=None, **kwargs):
        """
        Plots the 2D projection of the trajectory of all blocks.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to plot on.
            **kwargs: Additional arguments passed to `ax.plot`.

        Returns:
            matplotlib.axes.Axes: The axis with plotted trajectories.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Please install it with `pip install matplotlib`."
            ) from e

        if ax is None:
            fig, ax = plt.subplots()

        x, y = self.__call__()
        ax.plot(x, y, **kwargs)
        return ax


    def reasonStopped(self):
        """
        Returns:
            list of str: Reason each block stopped (default: "Stopped").
        """
        return ["Stopped"] * self.numberOfBlocks


    def __call__(self, numPoints=100):
        """
        Computes interpolated trajectories between start and stop time.

        Args:
            numPoints (int): Number of interpolation points.

        Returns:
            np.ndarray: Interpolated positions.
        """
        time = np.linspace(self.startTime, self.stopTime, numPoints)
        return self.position(time)
    

    def writeTXT(self, basename, time_step, compress=True):
        """
        Exports simulation results to plain-text files in PyRockfall-compatible format.

        Args:
            basename (str or list): Base file name or directory.
            time_step (float): Temporal resolution of output.
            compress (bool): Whether to use gzip compression.

        Returns:
            list of str: Paths to output files.
        """
        ind = '  '
        axes = ('x', 'y', 'z')
        ti = self.startTime
        tf = self.stopTime
        num_time_steps = np.ceil((tf - ti) / time_step).astype(int)
        time = np.linspace(self.startTime, self.stopTime, num_time_steps.max())
        x = self.position(time)
        v, w = self.velocity(time, angVel=True)
        reason = self.reasonStopped()
        a = self.acceleration(time)
        kt = self.mass * np.sum(v**2, axis=0) / 2
        kr = self.inertia * w**2 / 2
        u = self.mass * self.gravity * (x[1, :, :] - self.floor)
        e = kt + kr + u
        num_dim, num_time_steps, num_samples = x.shape
        axes = axes[:num_dim]
        files = []
        name_count = -1
        prev_name = ''
        for s in range(num_samples):
            if isinstance(basename, str):
                filename = basename
            else:
                filename = basename[s]
            if filename != prev_name:
                name_count = -1
                prev_name = filename
            name_count += 1
            ext = 'txt.gz' if compress else 'txt'
            if os.path.isdir(filename):
                filename = os.path.join(filename, f'{name_count}.{ext}')
            else:
                filename = f'{filename}{name_count}.{ext}'
            f = gzip.open(filename, 'wt') if compress else open(filename, 'w')
            files.append(filename)
            f.write('PyRockfall Results File:\n')
            f.write(ind+'version: 2.1\n')
            f.write('\n')
            f.write(ind+'path results data:\n')
            f.write(ind+'complete: yes\n')
            f.write(ind+f'end reason: {reason[s]}\n')
            f.write(ind+f'Total events: {self.positionHistory.shape[0]}\n')
            if isinstance(self.mass, Real):
                f.write(ind+f'rock mass: {self.mass}\n')
            else:
                f.write(ind+f'rock mass: {self.mass[s]}\n')
            if isinstance(self.inertia, Real):
                f.write(ind+f'rock inertia: {self.inertia}\n')
            else:
                f.write(ind+f'rock inertia: {self.inertia[s]}\n')
            f.write(ind+f'rock start time: {ti[s]}\n')
            for id, axis in enumerate(axes):
                f.write(ind+f'rock start Position {axis}: {self.positionHistory[0, id, s]}\n')
            f.write(ind+f'rock start rotation: 0.0\n')
            for id, axis in enumerate(axes):
                f.write(ind+f'rock start velocity {axis}: {self.velocityHistory[0, id, s]}\n')
            if len(axes) == 2:
                f.write(ind+f'rock start angular velocity: {self.angularVelocityHistory[0, 0, s]}\n')
            else:
                for id, axis in enumerate(axes):
                    f.write(ind+f'rock start angular velocity {axis}: {self.angularVelocityHistory[0, id, s]}\n')
            f.write(ind+'NOTES: Surface Type: 0-SEGMENT, 1-CORNER, 2-BARRIER, 5-FOREST BOUNDARY, 6/7-BERM, 8/9-INF. BERM\n')
            txt_pos = ', '.join([f'position {a}' for a in axes])
            txt_vel = ', '.join([f'velocity {a}' for a in axes])
            txt_acc = ', '.join([f'acceleration {a}' for a in axes])
            f.write(ind+f'time, {txt_pos}, rotation, {txt_vel}, angular velocity, {txt_acc}, angular acceleration, translational energy, rotational energy, potential energy, total energy, bounce height\n')
            if ti[s] < tf[s]:
                zeros_col = np.zeros((num_time_steps, 1))
                data_matrix = np.column_stack([
                    time[:, s],    # time
                    x[:, :, s].T,  # position x/y/z
                    zeros_col,     # rotation
                    v[:, :, s].T,  # velocity x/y/z
                    w[:, s],       # angular velocity
                    a[:, :, s].T,  # acceleration x/y/z
                    zeros_col,     # angular acceleration
                    kt[:, s],      # translational energy
                    kr[:, s],      # rotational energy
                    u[:, s],       # potential energy
                    e[:, s],       # total energy
                    zeros_col      # bounce height
                ])                
                # Format all rows with 9 decimal places and join them
                lines = [ind + ','.join(f'{val:.9f}' for val in row) + '\n' for row in data_matrix]
                f.writelines(lines)
            f.close()
        return files
    
    def readTXT(self, basename):
        """
        Imports trajectories from TXT or TXT.GZ files.

        Args:
            basename (str or list): File path or prefix.

        Returns:
            list of str: Files successfully read.
        """
        ind = '  '
        if isinstance(basename, str):
            basename = [basename]
        files = []
        for name in basename:
            name_count = -1
            while True:
                name_count += 1
                if os.path.isdir(name):
                    # If name is a directory, look for files in that directory
                    filename = os.path.join(name, f'{name_count}')
                else:
                    filename = f'{name}{name_count}'
                for ext in ('txt', 'txt.gz'):
                    if os.path.isfile(filename+f'.{ext}'):
                        filename = filename+f'.{ext}'
                        break
                else:
                    # If no file found, break the loop
                    break
                files.append(filename)
        if len(files) == 0:
            return []
        files_to_remove = []
        # Read the files
        time_history = []
        position_history = []
        velocity_history = []
        angular_velocity_history = []
        acceleration_history = []
        mass = []
        inertia = []
        floor = []
        for filename in files:
            if filename.endswith('.gz'):
                with gzip.open(filename, 'rt') as f:
                    lines = f.readlines()
            else:
                with open(filename, 'r') as f:
                    lines = f.readlines()
            # Extract data from the file
            # Initialize variables
            time = []
            positions = []
            velocities = []
            angular_velocities = []
            accelerations = []
            mass_value = None
            inertia_value = None
            floor_value = []
            has_time_series = False
            # Read the file line by line
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                # Check for specific keywords and extract data accordingly
                if line.startswith(ind+'rock mass:'):
                    mass_value = float(line.split(':')[1].strip())
                elif line.startswith(ind+'rock inertia:'):
                    inertia_value = float(line.split(':')[1].strip())
                if has_time_series:
                    # Read time series data
                    parts = line.split(',')
                    ndim = 3 if len(parts) > 15 else 2
                    time.append(float(parts[0].strip()))
                    positions.append([float(p.strip()) for p in parts[1:ndim+1]])
                    v0 = 2 + ndim
                    w = v0 + ndim
                    velocities.append([float(v.strip()) for v in parts[v0:w]])
                    angular_velocities.append(float(parts[w].strip()))
                    accelerations.append([float(a.strip()) for a in parts[w+1:w+1+ndim]])
                    if mass_value is None:
                        mass_value = []
                    if inertia_value is None:
                        inertia_value = []
                    if isinstance(mass_value, list):
                        id = 13 if ndim == 3 else 10
                        kt = float(parts[id].strip())
                        v_2 = np.sum(np.array(velocities[-1])**2)
                        if v_2 > 0:
                            mass_value.append(2*kt/v_2)
                    if isinstance(inertia_value, list):
                        id = 14 if ndim == 3 else 11
                        kr = float(parts[id].strip())
                        w_2 = angular_velocities[-1]**2
                        if w_2 > 0:
                            inertia_value.append(2*kr/w_2)
                    id = 15 if ndim == 3 else 12
                    U = float(parts[id].strip())
                    if isinstance(mass_value, list):
                        h = U / (self.gravity * mass_value[-1])
                    else:
                        h = U / (self.gravity * mass_value)
                    floor_value.append(h - positions[-1][1])                    
                elif line.startswith(ind+'time, position'):
                    has_time_series = True
            # End of file processing
            if len(time) == 0:
                print(f'Warning: No time series data found in {filename}. Skipping this file.')
                files_to_remove.append(filename)
                continue
            time_history.append(time)
            position_history.append(positions)
            velocity_history.append(velocities)
            angular_velocity_history.append(angular_velocities)
            acceleration_history.append(accelerations)
            if isinstance(mass_value, list):
                mass_value = np.mean(mass_value)
            mass.append(mass_value)
            if isinstance(inertia_value, list):
                inertia_value = np.mean(inertia_value)
            inertia.append(inertia_value)
            floor.append(np.mean(floor_value))
        # Check if any file was read
        if len(time_history) == 0:
            print('Warning: No valid files found.')
            return []
        # Store the data in the object
        self._time_history = np.transpose(np.asarray(time_history), (1, 0))
        self._position_history = np.transpose(np.asarray(position_history), (1, 2, 0))
        self._velocity_history = np.transpose(np.asarray(velocity_history), (1, 2, 0))
        self._angular_velocity_history = np.transpose(np.asarray(angular_velocity_history), (1, 0))
        self._acceleration_history = np.transpose(np.asarray(acceleration_history), (1, 2, 0))
        self.mass = np.asarray(mass)
        self.inertia = np.asarray(inertia)
        self.floor = np.asarray(floor)
        # Remove files that were not read
        for filename in files_to_remove:
            files.remove(filename)
        return files

    def writeNPZ(self, basename, time_step=None):
        """
        Saves the trajectory data to a compressed .npz file.

        Args:
            basename (str): Output path (without .npz).
            time_step (float, optional): Resample time step.

        Returns:
            str: Path to the saved file.
        """
        if os.path.isdir(basename):
            basename = os.path.join(basename, 'trajectories')
        if basename.endswith('.npz'):
            basename = basename.split('.npz')[0]
        if time_step is not None:
            num_time_steps = np.ceil((self.stopTime - self.startTime) / time_step).astype(int)
            t = np.linspace(self.startTime, self.stopTime, num_time_steps.max())
            x = self.position(t)
            v, w = self.velocity(t, angVel=True)
            a = self.acceleration(t)
        else:
            t = self.timeHistory
            x = np.transpose(self.positionHistory, (1, 0, 2))
            v = np.transpose(self.velocityHistory, (1, 0, 2))
            w = np.transpose(self.angularVelocityHistory, (1, 0, 2))
            a = np.transpose(self.accelerationHistory, (1, 0, 2))
        np.savez_compressed(
            basename,
            time_history=t,
            position_history=x,
            velocity_history=v,
            angular_velocity_history=w,
            acceleration_history=a,
            mass=self.mass,
            inertia=self.inertia,
            floor=self.floor,
            gravity=self.gravity
        )
        return basename + '.npz'
    
    def readNPZ(self, filename):
        """
        Loads trajectory data from a .npz file.

        Args:
            filename (str): Path to the NPZ file.
        """
        data = np.load(filename)
        self._time_history = data['time_history']
        self._position_history = np.transpose(data['position_history'], (1, 0, 2))
        self._velocity_history = np.transpose(data['velocity_history'], (1, 0, 2))
        self._angular_velocity_history = data['angular_velocity_history']
        self._acceleration_history = np.transpose(data['acceleration_history'], (1, 0, 2))
        self.mass = data['mass']
        self.inertia = data['inertia']
        self.floor = data['floor']
        self.gravity = data['gravity']
    