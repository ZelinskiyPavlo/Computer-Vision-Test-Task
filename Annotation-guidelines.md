# Annotation Guidelines — how to annotate training data

Data: aerial (drone) urban video with car traffic.

Task: 2D object detection, with single class `vehicle`.

Wording:
+ `MUST` — hard rule.
+ `SHOULD` — default unless there is a clear reason.
+ `MAY` — optional.

# What to annotate
A vehicle is a motorized road vehicle with four or more wheels, together with anything physically attached to it and moving with it (load, roof box, attached trailer).

Must always annotate:
+ Sedan, hatchback, SUV
+ Van, minivan, minibus, pickup
+ Box truck, City bus, tow truck, garbage truck
+ Police, ambulance, fire engine

Vehicle can be in any moving state: moving, stopped at a light, parked on the street, in a driveway, etc.

# What NOT to annotate

Must not annotate:
+ Reflections of vehicles (glass facades, water, wet asphalt).
+ Shadows of vehicles, including a shadow whose vehicle is hidden or out of frame.
+ Pictures of vehicles on billboards, screen, buildings.
+ Vehicle-like objects: dumpsters, containers on the ground, rooftop AC units.
+ Isolated parts of the vehicle: a detached door, a single wheel.
+ Vehicle you cannot recognize from the frame alone.
+ Fully hidden vehicles, even if you know from the video they are there.

# Bounding box rules
The box is the smallest axis-aligned rectangle enclosing all visible pixels of the vehicle.

Bounding box must include:
+ Body, roof, windshield
+ Open doors, open trunk, or hood
+ Attached load, roof box, roof rack
+ Attached trailer

Bounding box should exclude:
+ Tiny vehicle parts that sticks out, e.g. side mirrors, antennas
  + Including them will add small size inconsistency, which will distort small bounding box sizes
+ Headlight beams, glare, reflections on the road
+ Motion trails (due to low image quality), compression ghosting

## Rotated vehicles
Draw the axis-aligned rectangle that encloses the whole rotated vehicle. Do not shrink it to reduce background.

### Precision
Each edge SHOULD ideally be within ±3 px of the outermost visible vehicle pixel at native resolution.

### Motion blur
+ If blur makes the vehicle unrecognizable from a single frame, do not label such frame.

### Single box, for one vehicle
+ Boxes of different vehicles MAY overlap (dense traffic, oblique views).
  + But you MUST not draw ONE box around several vehicles.
+ If an occluder (pole, wire, branch) splits a vehicle into fragments and you are confident they belong to one vehicle, draw one box spanning all fragments.