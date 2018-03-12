import numpy as np


def data_1d(num_samples):
    x = np.float32(np.random.uniform(-10.5, 10.5, (1, num_samples))).T
    r = np.float32(np.random.normal(size=(num_samples, 1)))
    y = np.float32(np.sin(0.75 * x) * 7.0 + x * 0.5 + r * 1.0)
    return {'input': x, 'output': y}


def data_2d(num_samples):
    L1, L2 = 0.8, 0.2
    theta1 = np.float32(np.random.uniform(0.3, 1.2, (num_samples, 1)))
    theta2 = np.float32(np.random.uniform(np.pi / 2, 3 * np.pi / 2, (num_samples, 1)))
    joint_data = np.concatenate([theta1, theta2], axis=1)
    x1 = np.float32(L1 * np.cos(theta1) - L2 * np.cos(theta1 + theta2))
    x2 = np.float32(L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2))
    end_data = np.concatenate([x1, x2], axis=1)

    # invert training data
    temp_data = joint_data
    x = end_data
    y = temp_data
    return {'input': x, 'output': y}