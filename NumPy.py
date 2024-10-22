import numpy as np
a = np.array([1,2,3,4,5])
b = np.array((1,6,3,4,9))
print(a.ndim)
print(type(a))

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.ndim)

arr = np.array([1, 2, 3, 4, 5])
print(arr)

arr = np.arange(0, 10, 2)
print(arr)

arr = np.arange(1, 10)
reshaped_arr = arr.reshape(3, 3)
print(reshaped_arr)


random_num = np.random.rand()
print(random_num)

random_arr = np.random.randint(1, 10, size=(2, 3))
print(random_arr)

