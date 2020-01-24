Sample Space
============


.. code-block:: python

    from sspace import Space, either, eq
    import json

    space = Space()

    optim = space.categorical('optimizer', ['sgd', 'adam'])

    sgd_lr = space.loguniform('optimizer.lr', 1, 2, quantization=0.01)
    sgd_lr.enable_if(either(eq(optim, 'adam'), eq(optim, 'sgd')))
    sgd_lr.forbid(eq(sgd_lr, 1))

    space.config_space()

    for sample in space.sample(2):
        print(sample)

    print(json.dumps(space.serialize(), indent=2))
