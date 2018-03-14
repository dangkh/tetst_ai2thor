import ai2thor.controller
import random

import matplotlib.pyplot as plt
controller = ai2thor.controller.Controller()
controller.start()

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430

controller.reset('FloorPlan28')
controller.step(dict(action='Initialize', gridSize=0.25))


for i in range(1):
	event = controller.step(dict(action='RotateLeft'))
	event = controller.step(dict(action='RotateLeft'))
	event = controller.step(dict(action='MoveRight'))
	event = controller.step(dict(action='MoveRight'))
	event = controller.step(dict(action='MoveRight'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='MoveAhead'))
	event = controller.step(dict(action='RotateLeft'))
	event = controller.step(dict(action='MoveAhead'))

	img = event.frame
	
	print event.metadata["lastActionSuccess"]
	tmp = event.metadata["objects"]
	x = tmp[19]['visible']
	for idx, i in enumerate(tmp):
		print idx
		print i
	print x
	print tmp[43]


	
	
# y - shape (width, height, channels), channels are in RGB order

# action = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft',
#         'RotateRight', 'RotateLeft']

# import general_env

# myclass = general_env.environment()
# for i in range(10):
# 	print "/////////////////////////////////////////////"
# 	print i
# 	state, reward, done = myclass.step(5)
# 	print "output will be: " + str(done)
# 	tmp = myclass.event.metadata["lastActionSuccess"]
# 	print "check myself"
# 	print tmp
# 	print not tmp
# 	myclass.env_reset()
