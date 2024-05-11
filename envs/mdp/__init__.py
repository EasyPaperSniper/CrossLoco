# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with implementation of manager terms.

The functions can be provided to different managers that are responsible for the
different aspects of the MDP. These include the observation, reward, termination,
actions, randomization and curriculum managers.

The terms are defined under the ``envs`` module because they are used to define
the environment. However, they are not part of the environment directly, but
are used to define the environment through their managers.

"""


from .observations import *  # noqa: F401, F403
from .randomizations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *
