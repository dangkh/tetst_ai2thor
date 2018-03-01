import gym
import ai2thor.controller


def set_name_scene(name_scene, type_env):
  tmp = ['FloorPlan28', 'CartPole-v0']
  if name_scene == None:
    return tmp[type_env]
  else:
    return name_scene

def check(info):
  return not info.metadata['lastActionSuccess']



class environment():
  def __init__(self, default= 0, name_scene= None):
    # set type of environment
    # default: 0 = ai2thor; 1 = gym
    self.env = default
    self.name_scene = set_name_scene(name_scene, self.env)
    if default == 0:
      self.action_range = 6
      self.action = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft',
        'RotateRight', 'RotateLeft']
      self.controller = ai2thor.controller.Controller()
      self.controller.start()
      self.controller.reset(self.name_scene)
      self.event = self.controller.step(dict(action='Initialize', gridSize=0.5))
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

  def step(self, action_number= None):
    done = False
    reward = -99999
    state = None
    print action_number
    if self.env == 0:
      info = self.controller.step(dict(action= action_number))
      state = info.frame

      done = check(info)
      if done:
        reward = -1000
      elif action_number <=4 :
        reward = 1
      else:
        reward = 0

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
      self.event = self.controller.step(dict(action='Initialize', gridSize=0.5))
    else:
      self.controller.reset()

  def get_observation(self):
    if self.env == 0:
      return self.event.frame
    else:
      return self.controller



