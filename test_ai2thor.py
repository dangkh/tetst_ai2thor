import ai2thor.controller
controller = ai2thor.controller.Controller()
controller.start()

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430

controller.reset('FloorPlan28')
controller.step(dict(action='Initialize', gridSize=0.25))

event = controller.step(dict(action='MoveAhead'))

# y - shape (width, height, channels), channels are in RGB order
tmp = event.frame
print tmp.shape
