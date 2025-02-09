# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains an agent to locomote to a target location."""

from typing import Tuple

import brax
from brax.envs import env
from brax.physics import math
import jax
import jax.numpy as jnp


class Fetch(env.Env):
  """Fetch trains a dog to run to a target location."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.target_idx = self.sys.body_idx['Target']
    self.torso_idx = self.sys.body_idx['Torso']
    self.target_radius = 2
    self.target_distance = 15

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = qp.replace(pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jnp.zeros(3)
    metrics = {
        'hits': zero,
        'weightedHits': zero,
        'movingToTarget': zero,
        'torsoIsUp': zero,
        'torsoHeight': zero
    }
    info = {'rng': rng}
    return env.State(qp, obs, reward, done, metrics, info)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # small reward for torso moving towards target
    torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
    target_rel = qp.pos[self.target_idx] - qp.pos[self.torso_idx]
    target_dist = jnp.linalg.norm(target_rel)
    target_dir = target_rel / (1e-6 + target_dist)
    moving_to_target = .1 * jnp.dot(torso_delta, target_dir)

    # small reward for torso being up
    up = jnp.array([0., 0., 1.])
    torso_up = math.rotate(up, qp.rot[self.torso_idx])
    torso_is_up = .1 * self.sys.config.dt * jnp.dot(torso_up, up)

    # small reward for torso height
    torso_height = .1 * self.sys.config.dt * qp.pos[0, 2]

    # big reward for reaching target and facing it
    fwd = jnp.array([1., 0., 0.])
    torso_fwd = math.rotate(fwd, qp.rot[self.torso_idx])
    torso_facing = jnp.dot(target_dir, torso_fwd)
    target_hit = jnp.where(target_dist < self.target_radius, 1.0, 0.0)
    weighted_hit = target_hit * torso_facing

    reward = torso_height + moving_to_target + torso_is_up + weighted_hit

    state.metrics.update(
        hits=target_hit,
        weightedHits=weighted_hit,
        movingToTarget=moving_to_target,
        torsoIsUp=torso_is_up,
        torsoHeight=torso_height)

    # teleport any hit targets
    rng, target = self._random_target(state.info['rng'])
    target = jnp.where(target_hit, target, qp.pos[self.target_idx])
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = qp.replace(pos=pos)
    state.info.update(rng=rng)
    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Egocentric observation of target and the dog's body."""
    torso_fwd = math.rotate(jnp.array([1., 0., 0.]), qp.rot[self.torso_idx])
    torso_up = math.rotate(jnp.array([0., 0., 1.]), qp.rot[self.torso_idx])

    v_inv_rotate = jax.vmap(math.inv_rotate, in_axes=(0, None))

    pos_local = qp.pos - qp.pos[self.torso_idx]
    pos_local = v_inv_rotate(pos_local, qp.rot[self.torso_idx])
    vel_local = v_inv_rotate(qp.vel, qp.rot[self.torso_idx])

    target_local = pos_local[self.target_idx]
    target_local_mag = jnp.reshape(jnp.linalg.norm(target_local), -1)
    target_local_dir = target_local / (1e-6 + target_local_mag)

    pos_local = jnp.reshape(pos_local, -1)
    vel_local = jnp.reshape(vel_local, -1)

    contact_mag = jnp.sum(jnp.square(info.contact.vel), axis=-1)
    contacts = jnp.where(contact_mag > 0.00001, 1, 0)

    return jnp.concatenate([
        torso_fwd, torso_up, target_local_mag, target_local_dir, pos_local,
        vel_local, contacts
    ])

  def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a target location in a random circle on xz plane."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    dist = self.target_radius + self.target_distance * jax.random.uniform(rng1)
    ang = jnp.pi * 2. * jax.random.uniform(rng2)
    target_x = dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = 1.0
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target


_SYSTEM_CONFIG = """
bodies {
  name: "Torso"
  colliders {
    box {
      halfsize { x: 0.75 y: 0.25 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Shoulders"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.75 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Hips"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.75 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Right Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Right Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}

bodies {
  name: "Front Left Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Left Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Right Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Right Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Left Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Left Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Ground"
  colliders { plane {} }
  frozen { all: true }
}
bodies {
  name: "Target"
  colliders { sphere { radius: 2 }}
  frozen { all: true }
}
joints {
  name: "Torso_Shoulders"
  angle_limit { min: -60 max: 60 }
  parent_offset { x: 1.0 }
  child_offset {}
  parent: "Torso"
  child: "Shoulders"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Torso_Hips"
  angle_limit { min: -60 max: 60 }
  parent_offset { x: -1.0 }
  child_offset {}
  parent: "Torso"
  child: "Hips"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Shoulders_Front Right Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.875 }
  child_offset { z: 0.375 }
  parent: "Shoulders"
  child: "Front Right Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Front Right Upper_Front Right Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Front Right Upper"
  child: "Front Right Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Shoulders_Front Left Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.875 }
  child_offset { z: 0.375 }
  parent: "Shoulders"
  child: "Front Left Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Front Left Upper_Front Left Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Front Left Upper"
  child: "Front Left Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Hips_Back Right Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.875 }
  child_offset { z: 0.375 }
  parent: "Hips"
  child: "Back Right Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Back Right Upper_Back Right Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Back Right Upper"
  child: "Back Right Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Hips_Back Left Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.875 }
  child_offset { z: 0.375 }
  parent: "Hips"
  child: "Back Left Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Back Left Upper_Back Left Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Back Left Upper"
  child: "Back Left Lower"
  stiffness: 5000.0
  angular_damping: 35
}
actuators {
  name: "Torso_Shoulders"
  torque {}
  joint: "Torso_Shoulders"
  strength: 300.0
}
actuators {
  name: "Torso_Hips"
  torque {}
  joint: "Torso_Hips"
  strength: 300.0
}
actuators {
  name: "Shoulders_Front Right Upper"
  torque {}
  joint: "Shoulders_Front Right Upper"
  strength: 300.0
}
actuators {
  name: "Front Right Upper_Front Right Lower"
  torque {}
  joint: "Front Right Upper_Front Right Lower"
  strength: 300.0
}
actuators {
  name: "Shoulders_Front Left Upper"
  torque {}
  joint: "Shoulders_Front Left Upper"
  strength: 300.0
}
actuators {
  name: "Front Left Upper_Front Left Lower"
  torque {}
  joint: "Front Left Upper_Front Left Lower"
  strength: 300.0
}
actuators {
  name: "Hips_Back Right Upper"
  torque {}
  joint: "Hips_Back Right Upper"
  strength: 300.0
}
actuators {
  name: "Back Right Upper_Back Right Lower"
  torque {}
  joint: "Back Right Upper_Back Right Lower"
  strength: 300.0
}
actuators {
  name: "Hips_Back Left Upper"
  torque {}
  joint: "Hips_Back Left Upper"
  strength: 300.0
}
actuators {
  name: "Back Left Upper_Back Left Lower"
  torque {}
  joint: "Back Left Upper_Back Left Lower"
  strength: 300.0
}
friction: 0.6
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "Front Right Lower"
  second: "Ground"
}
collide_include {
  first: "Front Left Lower"
  second: "Ground"
}
collide_include {
  first: "Back Right Lower"
  second: "Ground"
}
collide_include {
  first: "Back Left Lower"
  second: "Ground"
}
dt: 0.02
substeps: 4
"""
