actions = []
'''
actions:
move table x
move table y
but which table?
how can I get an action from an image that applies to the json?
one option: could it be three dimensional? <item,target_x,target_y>
or: it could be <which item, which direction>
or: <item_x,item_y,which direction of four, (and how much?)> problem: i need to learn on the objects, not on the image locations? what if we learn not to move anything in one particular spot?
---> if there is no item there, then automatically return a huge negative reward

'''