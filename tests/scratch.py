from sspace import Space


space = Space()
space.uniform('lr', 0, 1)
space.ordinal('epoch', [1, 2, 3])


s1 = space.sample(seed=0)
s2 = space.sample(seed=1)
s1p = space.sample(seed=0)


print(s1)
print(s2)
print(s1p)

# [OrderedDict([('epoch', 1), ('optimizer', 'adam')])]
# [OrderedDict([('epoch', 2), ('optimizer', 'adam')])]
# [OrderedDict([('epoch', 1), ('optimizer', 'adam')])]
