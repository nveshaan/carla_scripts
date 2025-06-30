#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import numpy as np

def vector(location_1, location_2):
        """
        Returns the unit vector from location_1 to location_2
        location_1, location_2    :   carla.Location objects
        """
        x = location_2.x - location_1.x
        y = location_2.y - location_1.y
        z = location_2.z - location_1.z
        norm = np.linalg.norm([x, y, z])

        return [x/norm, y/norm, z/norm]