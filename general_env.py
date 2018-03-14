import gym
import ai2thor.controller
import matplotlib.pyplot as plt

def set_name_scene(name_scene, type_env):
  tmp = ['FloorPlan28', 'CartPole-v0']
  if name_scene == None:
    return tmp[type_env]
  else:
    return name_scene


class environment():
  def __init__(self, default= 0, name_scene= None):
    # set type of environment
    # default: 0 = ai2thor; 1 = gym
    print "Initialize environment"
    self.env = default
    self.name_scene = set_name_scene(name_scene, self.env)
    if default == 0:
      self.action_range = 6
      self.action = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft',
        'RotateRight', 'RotateLeft']
      self.controller = ai2thor.controller.Controller()
      self.controller.start()
      self.controller.reset(self.name_scene)
      self.event = self.controller.step(dict(action='Initialize', gridSize=0.05))
      self.angle = 0
    else:
      self.controller = gym.make(self.name_scene)
      self.controller.reset()
      self.action_range = self.controller.action_space
      self.action = None
    print "Initialized an environment"

  def render(self):
    #only for gym
    if self.env == 1:
      self.controller.render()

  def step(self, action_number= 0):
    done = False
    reward = -99999
    state = None
    if self.env == 0:
      print self.action[action_number]
      # info = self.controller.step(dict(action= self.action[action_number]))
      if action_number < 4:
        info = self.controller.step(dict(action= self.action[action_number]))
      elif action_number == 4:
        self.angle += 30
        info = self.controller.step(dict(action= 'Rotate', rotation=self.angle))
      else:
        self.angle -= 30
        info = self.controller.step(dict(action= 'Rotate', rotation=self.angle))

      state = info.frame
      self.event = info

      done = not self.event.metadata["lastActionSuccess"]
      if done:
        reward = -1000
      elif action_number <4 :
        reward = -0.5
      else:
        reward = -0.2

      done = False
      if self.event.metadata['objects'][43]['visible']:
        reward = 1000
        done = True

      return state, reward, done
    else:
      state, reward, done, info = self.controller.step(action_number)

    return state, reward, done

  def print_information(self):

    print "action_range", self.action_range
    print "action_list", self.action
    if self.env == 0:
      print "Environment using Ai2thor"
    else:
      print "Environment using OpenAi Gym"

  def env_reset(self):
    if self.env == 0:
      self.controller.reset(self.name_scene)
      self.event = self.controller.step(dict(action='Initialize', gridSize=0.25))
    else:
      self.controller.reset()

  def get_observation(self):
    if self.env == 0:
      return self.event.frame
    else:
      return self.controller



