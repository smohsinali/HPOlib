import numpy as np
from benchmarks import svm_on_grid

b = svm_on_grid.SVM()

values = []

cs = b.get_configuration_space()

for i in range(20):
    configuration = cs.sample_configuration()
    # Configuration does not yet implement __len__, so we have to call
    # get_dictionary for now!
    rval = b.evaluate_dict(configuration.get_dictionary())
    loss = rval['function_value']

    values.append(loss)

# print(values)
print(np.min(values))