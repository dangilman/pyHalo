import numpy as np

a_fit = np.array([1.634, 1.651, 1.669, 1.686, 1.703, 1.707, 1.698, 1.689, 1.679, 1.67, 1.629, 1.654, 1.697, 1.74, 1.784, 1.798, 1.782, 1.767, 1.751, 1.74, 1.732, 1.716, 1.736, 1.758, 1.781, 1.766, 1.713, 1.66, 1.613, 1.628, 1.644, 1.622, 1.594, 1.566, 1.538, 1.547, 1.594, 1.642, 1.689, 1.721, 1.643, 1.617, 1.59, 1.564, 1.537, 1.546, 1.604, 1.672, 1.739, 1.806, 1.683, 1.673, 1.662, 1.649, 1.623, 1.643, 1.711, 1.717, 1.711, 1.704, 1.686, 1.676, 1.665, 1.656, 1.647, 1.654, 1.679, 1.692, 1.686, 1.679, 1.65, 1.641, 1.632, 1.623, 1.63, 1.628, 1.618, 1.643, 1.668, 1.693, 1.627, 1.615, 1.624, 1.633, 1.642, 1.64, 1.627, 1.615, 1.602, 1.608, 1.672, 1.652, 1.632, 1.612, 1.592, 1.589, 1.604, 1.618, 1.633, 1.648, 1.826, 1.758, 1.69, 1.659, 1.676, 1.68, 1.671, 1.708, 1.778, 1.849, 1.806, 1.7, 1.593, 1.535, 1.571, 1.694, 1.796, 1.853, 1.841, 1.83, 1.886, 1.871, 1.763, 1.647, 1.577, 1.672, 1.698, 1.645, 1.662, 1.753, 1.887, 1.864, 1.829, 1.702, 1.575, 1.542, 1.695, 1.847, 1.902, 1.949, 1.818, 1.79, 1.736, 1.681, 1.619, 1.628, 1.737, 1.862, 1.987, 2.056, 1.859, 1.833, 1.807, 1.785, 1.758, 1.779, 1.846, 1.878, 1.916, 1.944, 1.868, 1.842, 1.816, 1.791, 1.782, 1.79, 1.814, 1.819, 1.797, 1.797, 1.905, 1.837, 1.776, 1.74, 1.731, 1.739, 1.763, 1.78, 1.786, 1.788, 1.886, 1.815, 1.744, 1.698, 1.699, 1.697, 1.692, 1.711, 1.736, 1.76, 1.92, 1.84, 1.759, 1.704, 1.684, 1.681, 1.695, 1.723, 1.76, 1.796, 2.182, 2.114, 2.046, 1.979, 1.911, 1.774, 1.845, 1.916, 1.986, 2.057, 2.097, 1.991, 1.886, 1.78, 1.73, 1.585, 1.669, 1.771, 1.873, 1.975, 2.152, 1.941, 1.825, 1.709, 1.59, 1.573, 1.658, 1.75, 1.841, 1.918, 2.19, 1.979, 1.853, 1.726, 1.599, 1.579, 1.699, 1.852, 2.004, 2.157, 2.136, 1.945, 1.891, 1.836, 1.775, 1.774, 1.899, 2.024, 2.148, 2.261, 2.152, 2.126, 2.1, 2.069, 2.015, 1.901, 2.026, 2.069, 2.096, 2.122, 2.244, 2.218, 2.192, 2.147, 2.091, 1.921, 1.931, 1.978, 2.004, 2.031, 2.3, 2.229, 2.152, 2.096, 2.04, 1.923, 1.946, 1.956, 1.966, 1.975, 2.283, 2.211, 2.14, 2.069, 2.003, 1.852, 1.876, 1.901, 1.926, 1.959, 2.313, 2.232, 2.151, 2.07, 1.99, 1.84, 1.877, 1.914, 1.951, 1.987, 3.021, 2.953, 2.885, 2.684, 2.227, 2.043, 2.131, 2.208, 2.279, 2.349, 2.745, 2.64, 2.541, 2.399, 1.991, 1.831, 1.912, 1.986, 2.087, 2.189, 2.701, 2.331, 2.191, 2.075, 1.901, 1.786, 1.871, 1.962, 2.021, 2.074, 2.713, 2.488, 2.361, 2.234, 1.915, 1.79, 1.835, 1.904, 2.057, 2.21, 2.778, 2.372, 2.318, 2.263, 2.081, 1.916, 2.016, 2.141, 2.266, 2.316, 2.8, 2.774, 2.748, 2.594, 2.204, 2.04, 2.1, 2.14, 2.166, 2.193, 2.975, 2.949, 2.923, 2.743, 2.294, 2.092, 2.184, 2.251, 2.277, 2.304, 3.071, 2.962, 2.883, 2.692, 2.243, 2.119, 2.225, 2.235, 2.245, 2.29, 3.07, 2.998, 2.928, 2.731, 2.257, 2.061, 2.177, 2.232, 2.257, 2.29, 3.131, 3.05, 2.97, 2.752, 2.275, 2.082, 2.174, 2.23, 2.267, 2.303, 3.86, 3.792, 3.407, 2.95, 2.494, 2.31, 2.398, 2.486, 2.571, 2.642, 3.394, 3.356, 2.996, 2.616, 2.235, 2.075, 2.156, 2.237, 2.317, 2.402, 3.249, 2.879, 2.74, 2.458, 2.113, 1.999, 2.072, 2.125, 2.178, 2.231, 3.236, 3.01, 2.803, 2.464, 2.126, 2.001, 2.046, 2.091, 2.136, 2.262, 3.421, 3.014, 2.96, 2.619, 2.23, 2.065, 2.126, 2.196, 2.321, 2.37, 3.491, 3.465, 3.127, 2.723, 2.333, 2.169, 2.23, 2.295, 2.353, 2.38, 3.673, 3.647, 3.31, 2.877, 2.428, 2.249, 2.34, 2.422, 2.481, 2.507, 3.802, 3.693, 3.306, 2.857, 2.408, 2.272, 2.379, 2.388, 2.472, 2.563, 3.857, 3.797, 3.398, 2.927, 2.455, 2.259, 2.375, 2.49, 2.588, 2.622, 3.949, 3.869, 3.464, 2.987, 2.51, 2.318, 2.409, 2.501, 2.583, 2.619, 5.438, 4.539, 4.083, 3.627, 3.17, 2.986, 3.074, 3.162, 3.25, 3.14, 4.91, 4.53, 4.15, 3.769, 3.129, 2.617, 2.698, 2.779, 2.888, 2.876, 4.001, 3.708, 3.263, 2.919, 2.575, 2.276, 2.33, 2.383, 2.436, 2.489, 3.992, 3.564, 3.225, 2.887, 2.548, 2.288, 2.334, 2.379, 2.424, 2.469, 4.155, 3.786, 3.388, 2.998, 2.609, 2.444, 2.505, 2.566, 2.512, 2.553, 4.315, 3.916, 3.534, 3.129, 2.721, 2.551, 2.617, 2.681, 2.729, 2.662, 4.53, 4.077, 3.628, 3.193, 2.786, 2.615, 2.681, 2.772, 2.864, 2.812, 4.691, 4.249, 3.8, 3.351, 2.89, 2.7, 2.791, 2.882, 2.984, 2.949, 4.855, 4.346, 3.874, 3.402, 2.931, 2.741, 2.851, 2.966, 3.082, 3.047, 4.957, 4.46, 3.983, 3.506, 3.029, 2.836, 2.928, 3.02, 3.111, 3.056, 10.04, 8.404, 6.768, 5.978, 5.521, 5.337, 5.425, 5.364, 4.924, 4.484, 7.223, 6.842, 6.462, 5.189, 4.743, 4.321, 4.402, 4.624, 4.354, 3.964, 5.945, 5.779, 5.168, 4.513, 4.053, 3.655, 3.513, 3.566, 3.546, 3.218, 5.508, 5.051, 4.598, 4.176, 3.838, 3.559, 3.604, 3.649, 3.438, 3.18, 5.569, 5.157, 4.742, 4.326, 3.93, 3.765, 3.743, 3.463, 3.183, 3.236, 5.709, 5.286, 4.87, 4.49, 4.124, 3.953, 3.978, 3.916, 3.636, 3.362, 6.06, 5.604, 5.143, 4.678, 4.212, 4.041, 4.166, 4.174, 3.884, 3.597, 6.229, 5.796, 5.347, 4.895, 4.446, 4.267, 4.358, 4.342, 4.054, 3.77, 6.539, 6.04, 5.514, 5.013, 4.542, 4.352, 4.462, 4.465, 4.18, 3.896, 6.742, 6.21, 5.677, 5.185, 4.708, 4.515, 4.607, 4.587, 4.285, 3.983, 14.642, 13.006, 11.37, 9.734, 8.098, 7.588, 7.147, 6.707, 6.267, 5.827, 9.051, 7.803, 7.338, 6.892, 6.447, 6.197, 5.928, 5.658, 5.388, 4.998, 8.016, 7.513, 6.857, 6.202, 5.575, 5.134, 4.953, 4.625, 4.296, 3.968, 7.017, 6.562, 6.108, 5.654, 5.2, 4.848, 4.665, 4.407, 4.149, 3.891, 6.94, 6.501, 6.085, 5.669, 5.253, 4.906, 4.626, 4.346, 4.066, 3.908, 7.15, 6.722, 6.294, 5.872, 5.456, 5.218, 4.938, 4.663, 4.393, 4.123, 7.501, 7.073, 6.645, 6.187, 5.726, 5.477, 5.187, 4.898, 4.628, 4.358, 7.839, 7.341, 6.884, 6.423, 5.963, 5.709, 5.419, 5.129, 4.841, 4.557, 8.202, 7.703, 7.205, 6.706, 6.18, 5.902, 5.614, 5.33, 5.045, 4.761, 8.526, 7.994, 7.462, 6.93, 6.398, 6.119, 5.817, 5.515, 5.213, 4.911, 39.397, 37.76, 33.365, 28.951, 24.537, 21.378, 19.474, 17.57, 15.676, 15.236, 28.56, 25.921, 23.282, 21.113, 19.608, 16.449, 16.076, 14.156, 11.785, 9.414, 18.525, 16.113, 13.194, 10.24, 8.45, 7.959, 7.576, 7.138, 6.7, 6.263, 12.825, 11.671, 10.553, 9.434, 8.653, 8.175, 7.673, 7.172, 6.778, 6.442, 11.265, 10.849, 10.188, 9.524, 8.577, 7.887, 7.479, 7.072, 6.788, 6.294, 10.655, 10.123, 9.564, 8.926, 8.47, 7.934, 7.634, 7.262, 6.867, 6.472, 10.731, 10.199, 9.667, 9.159, 8.683, 8.308, 7.964, 7.565, 7.17, 6.774, 11.088, 10.596, 10.103, 9.64, 9.188, 8.762, 8.362, 7.92, 7.511, 7.16, 11.494, 10.971, 10.519, 10.067, 9.615, 9.279, 8.915, 8.468, 8.022, 7.532, 11.963, 11.431, 10.954, 10.477, 10.0, 9.611, 9.232, 8.773, 8.314, 7.855, 51.375, 57.213, 63.05, 68.888, 74.725, 74.917, 71.545, 68.172, 64.8, 61.576, 43.239, 47.952, 50.935, 53.918, 56.901, 56.581, 53.759, 50.937, 48.114, 44.701, 26.043, 29.165, 32.396, 35.637, 38.879, 38.678, 36.293, 33.908, 31.775, 29.985, 54.42, 51.723, 43.201, 34.678, 26.156, 21.348, 20.104, 18.86, 17.616, 16.134, 127.515, 116.054, 86.408, 56.762, 27.117, 14.712, 14.116, 14.166, 14.215, 14.264, 97.535, 83.988, 70.441, 52.426, 22.78, 10.878, 10.927, 11.596, 12.391, 13.187, 47.952, 34.406, 20.859, 14.772, 11.06, 9.602, 10.4, 11.059, 11.854, 12.65, 23.545, 22.333, 19.404, 15.691, 11.979, 10.327, 9.895, 10.672, 11.47, 12.267, 20.261, 19.049, 17.837, 16.626, 15.208, 14.088, 13.656, 13.224, 12.791, 11.695, 27.076, 25.198, 23.32, 21.442, 19.564, 18.005, 16.482, 14.959, 13.436, 12.022, ])

b_fit = np.array([1471.817, 1478.277, 1484.738, 1491.199, 1497.66, 1504.774, 1512.543, 1520.312, 1528.08, 1535.849, 1416.615, 1436.835, 1490.338, 1543.842, 1597.345, 1610.251, 1582.559, 1554.866, 1527.174, 1524.571, 1501.176, 1493.299, 1530.217, 1571.597, 1612.976, 1602.387, 1539.827, 1477.268, 1421.559, 1434.619, 1482.635, 1469.262, 1451.645, 1434.028, 1416.412, 1421.034, 1447.896, 1474.758, 1501.619, 1522.467, 1473.683, 1459.498, 1445.313, 1431.128, 1415.357, 1419.979, 1452.31, 1489.345, 1526.379, 1563.414, 1487.923, 1482.609, 1477.295, 1470.474, 1456.289, 1467.714, 1504.749, 1498.305, 1482.961, 1467.618, 1485.98, 1480.666, 1475.352, 1472.377, 1470.821, 1482.024, 1505.985, 1515.11, 1499.767, 1484.423, 1472.584, 1471.028, 1469.472, 1467.916, 1471.608, 1471.334, 1469.046, 1493.007, 1516.969, 1540.93, 1473.815, 1459.79, 1463.969, 1468.148, 1472.327, 1472.054, 1467.329, 1462.603, 1457.878, 1467.381, 1492.156, 1470.605, 1449.054, 1427.503, 1405.952, 1402.87, 1418.255, 1433.641, 1449.026, 1464.412, 631.618, 631.408, 631.198, 633.815, 640.276, 647.391, 655.16, 658.525, 658.65, 658.776, 604.437, 604.059, 605.651, 616.812, 656.972, 706.623, 706.836, 686.708, 658.687, 656.084, 650.748, 642.871, 641.628, 641.045, 654.449, 721.807, 696.386, 633.827, 601.781, 602.064, 643.617, 630.244, 613.896, 612.946, 611.995, 602.812, 603.52, 604.227, 629.098, 655.96, 639.979, 626.314, 625.607, 624.899, 616.894, 621.516, 633.886, 634.761, 635.635, 671.277, 643.341, 642.817, 642.293, 638.231, 624.046, 635.471, 672.506, 652.668, 614.074, 610.372, 642.692, 642.168, 641.644, 640.613, 639.056, 650.259, 674.22, 670.253, 633.012, 612.622, 644.791, 645.275, 644.964, 643.8, 642.244, 653.446, 677.407, 687.366, 676.933, 652.279, 648.354, 646.986, 646.071, 638.134, 635.016, 634.742, 635.721, 643.493, 643.798, 643.821, 660.251, 658.408, 656.564, 646.366, 624.815, 621.733, 637.118, 643.913, 644.387, 644.861, 4.114, 3.904, 3.695, 3.485, 3.275, 3.186, 3.311, 3.436, 3.562, 3.687, 5.696, 5.318, 4.94, 4.562, 4.307, 4.181, 4.343, 4.556, 4.769, 4.981, 8.388, 7.769, 7.187, 6.604, 5.987, 5.819, 6.127, 6.41, 6.693, 6.905, 11.939, 11.225, 10.275, 9.324, 8.374, 8.022, 8.55, 9.258, 9.965, 10.673, 15.087, 14.348, 13.641, 12.933, 12.204, 12.166, 13.041, 13.915, 14.789, 15.079, 19.362, 18.838, 18.314, 17.758, 17.051, 16.769, 17.643, 17.717, 17.628, 17.538, 24.217, 23.693, 23.169, 22.426, 21.551, 20.466, 20.123, 20.208, 20.118, 20.029, 29.065, 27.697, 26.489, 25.614, 24.738, 24.211, 24.202, 23.859, 23.516, 23.173, 33.572, 32.204, 30.835, 29.467, 28.076, 26.86, 26.883, 26.906, 26.929, 27.272, 38.911, 37.067, 35.223, 33.379, 31.535, 30.372, 30.846, 31.32, 31.794, 32.268, 3.991, 3.781, 3.572, 3.328, 3.021, 2.915, 3.012, 3.128, 3.253, 3.378, 5.227, 4.849, 4.484, 4.284, 3.941, 3.733, 3.856, 3.98, 4.188, 4.401, 7.56, 6.879, 6.287, 5.705, 5.165, 5.239, 5.547, 5.83, 5.952, 6.045, 10.573, 9.899, 8.949, 7.998, 7.338, 7.125, 7.251, 7.504, 8.212, 8.919, 13.96, 13.169, 12.462, 11.755, 10.917, 10.51, 11.14, 12.014, 12.888, 12.508, 18.219, 17.694, 17.17, 16.388, 15.339, 14.932, 15.167, 15.215, 15.125, 15.035, 23.161, 22.637, 22.113, 20.93, 18.778, 17.729, 18.28, 18.589, 18.499, 18.41, 28.367, 26.656, 25.433, 24.117, 21.965, 21.574, 22.315, 21.971, 21.628, 21.67, 33.022, 31.653, 30.296, 28.356, 25.333, 24.305, 25.145, 25.45, 25.473, 25.816, 38.66, 36.817, 34.973, 32.588, 29.176, 27.849, 28.608, 29.181, 29.655, 30.129, 3.868, 3.658, 3.369, 3.061, 2.754, 2.649, 2.745, 2.842, 2.944, 3.07, 4.758, 4.504, 4.22, 3.857, 3.493, 3.286, 3.408, 3.531, 3.654, 3.82, 6.732, 6.05, 5.459, 4.999, 4.585, 4.659, 4.907, 4.999, 5.092, 5.184, 9.207, 8.533, 7.704, 7.073, 6.441, 6.229, 6.354, 6.479, 6.605, 7.165, 12.832, 12.042, 11.335, 10.336, 9.287, 8.88, 9.115, 9.443, 10.317, 9.936, 16.958, 16.434, 15.119, 13.702, 12.653, 12.246, 12.481, 12.839, 13.136, 13.046, 21.939, 21.415, 20.1, 18.197, 16.045, 15.245, 15.796, 16.283, 16.58, 16.49, 27.311, 25.6, 23.528, 21.376, 19.224, 18.639, 19.379, 19.036, 19.499, 20.05, 32.471, 31.21, 28.276, 25.475, 22.673, 21.645, 22.486, 23.326, 24.017, 24.36, 38.41, 36.566, 33.441, 30.029, 26.617, 25.29, 26.049, 26.808, 27.515, 27.989, 4.064, 3.63, 3.322, 3.015, 2.707, 2.602, 2.699, 2.796, 2.892, 2.881, 4.842, 4.479, 4.116, 3.752, 3.33, 3.125, 3.248, 3.37, 3.505, 3.544, 5.828, 5.376, 5.034, 4.62, 4.207, 3.943, 4.057, 4.149, 4.242, 4.334, 8.149, 7.708, 7.077, 6.445, 5.814, 5.435, 5.561, 5.686, 5.811, 5.937, 11.667, 10.958, 10.231, 9.183, 8.134, 7.727, 7.962, 8.197, 8.22, 8.386, 15.744, 15.054, 13.964, 12.485, 10.993, 10.439, 10.822, 11.191, 11.37, 11.369, 20.685, 19.219, 17.067, 15.326, 13.833, 13.279, 13.662, 14.234, 14.785, 14.871, 26.222, 24.724, 22.572, 20.42, 17.917, 16.792, 17.343, 17.894, 18.565, 18.759, 32.079, 29.647, 26.795, 23.994, 21.192, 20.171, 20.995, 21.836, 22.677, 22.87, 38.115, 35.669, 32.257, 28.845, 25.432, 24.106, 24.865, 25.624, 26.383, 26.411, 5.566, 4.919, 4.272, 3.868, 3.561, 3.456, 3.552, 3.568, 3.377, 3.186, 5.146, 4.782, 4.419, 3.875, 3.786, 3.698, 3.821, 3.979, 3.859, 3.689, 5.823, 5.492, 5.242, 5.001, 4.652, 4.335, 4.271, 4.364, 4.401, 4.21, 7.632, 7.282, 7.176, 6.922, 6.291, 5.891, 6.017, 6.142, 6.015, 5.841, 10.796, 9.934, 9.842, 9.749, 8.935, 8.528, 8.609, 8.217, 7.824, 8.083, 14.685, 14.551, 14.459, 13.59, 12.401, 11.847, 11.927, 11.862, 11.47, 10.997, 19.466, 19.17, 18.847, 17.364, 15.423, 14.869, 15.7, 16.102, 15.412, 14.749, 24.995, 24.589, 24.232, 22.597, 20.445, 19.645, 20.196, 20.396, 19.624, 18.738, 31.223, 30.665, 30.005, 27.338, 24.537, 23.516, 24.339, 24.692, 23.805, 22.918, 37.636, 36.803, 35.97, 33.287, 29.875, 28.548, 29.307, 29.515, 28.322, 27.129, 7.067, 6.42, 5.773, 5.126, 4.48, 4.254, 4.064, 3.873, 3.682, 3.491, 5.196, 4.636, 4.538, 4.448, 4.359, 4.329, 4.21, 4.09, 3.971, 3.801, 5.939, 5.67, 5.429, 5.187, 4.963, 4.779, 4.688, 4.497, 4.305, 4.114, 7.106, 6.826, 6.721, 6.615, 6.51, 6.368, 6.268, 6.095, 5.921, 5.748, 9.773, 9.425, 9.332, 9.24, 9.147, 8.905, 8.512, 8.12, 7.727, 7.687, 13.55, 13.39, 13.231, 13.103, 13.011, 12.575, 12.183, 11.722, 11.198, 10.674, 18.331, 18.172, 18.012, 17.701, 17.378, 16.842, 16.152, 15.474, 14.95, 14.426, 23.936, 23.378, 23.005, 22.682, 22.36, 21.672, 20.982, 20.292, 19.521, 18.634, 30.287, 29.729, 29.171, 28.613, 27.923, 26.717, 25.766, 24.88, 23.993, 23.106, 37.158, 36.325, 35.491, 34.658, 33.825, 32.618, 31.425, 30.232, 29.039, 27.847, 15.865, 15.218, 13.67, 12.117, 10.563, 9.428, 8.713, 7.997, 7.284, 7.094, 12.33, 11.352, 10.374, 9.586, 9.09, 7.955, 7.815, 6.983, 5.958, 4.933, 9.351, 8.435, 7.208, 5.958, 5.219, 5.003, 4.846, 4.724, 4.6, 4.467, 8.567, 8.156, 7.764, 7.372, 7.125, 6.97, 6.767, 6.564, 6.374, 6.219, 10.11, 10.017, 9.91, 9.802, 9.498, 9.276, 9.095, 8.915, 8.529, 8.318, 13.104, 13.141, 13.143, 12.991, 12.896, 12.752, 12.394, 12.059, 11.803, 11.56, 17.458, 17.496, 17.534, 17.626, 17.525, 17.018, 16.5, 16.196, 15.953, 15.71, 22.915, 23.04, 23.166, 22.879, 23.083, 23.038, 22.696, 22.198, 21.414, 20.661, 29.279, 28.527, 28.774, 29.027, 29.28, 29.206, 28.482, 27.924, 27.366, 26.349, 36.49, 35.657, 35.895, 36.14, 36.385, 35.911, 34.933, 34.174, 33.416, 32.658, 18.41, 20.771, 23.132, 25.493, 27.854, 28.068, 26.931, 25.793, 24.656, 23.561, 15.95, 17.982, 19.509, 21.036, 22.563, 22.602, 21.577, 20.552, 19.528, 18.365, 9.779, 11.597, 13.367, 15.133, 16.899, 17.036, 16.156, 15.276, 14.495, 13.852, 20.292, 19.879, 17.363, 14.847, 12.332, 10.883, 10.456, 10.029, 9.602, 9.082, 52.082, 48.328, 36.913, 25.498, 14.083, 9.372, 9.346, 9.782, 10.217, 10.652, 42.915, 37.371, 31.827, 24.574, 13.159, 8.755, 9.19, 10.392, 11.751, 13.11, 24.079, 18.535, 12.99, 10.997, 10.115, 10.609, 12.479, 13.911, 15.27, 16.629, 15.181, 16.839, 16.753, 15.871, 14.989, 14.775, 14.938, 16.78, 18.649, 20.519, 17.793, 19.45, 21.108, 22.766, 24.61, 25.394, 25.557, 25.719, 25.882, 24.38, 30.926, 33.188, 35.451, 37.714, 39.976, 39.701, 37.297, 34.894, 32.49, 30.253, ])