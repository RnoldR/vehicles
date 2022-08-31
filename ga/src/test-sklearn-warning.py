import sys
import warnings
import sklearn
from sklearn import datasets, linear_model,exceptions
import matplotlib.pyplot as plt 

#>>>Start: Create dummy data
blob = datasets.make_blobs(n_samples=100,centers=1)[0]
x = blob[:,0].reshape(-1,1)
# y needs to be integer for logistic regression
y = blob[:,1].astype(int)
plt.scatter(x,y)
#plt.show()
#<<<End: Create dummy data

#<<Create logistic regression. set max_iteration to a low number
max_iters = [2, 10, 100]

for max_iter in max_iters:
    print()
    print('*** max_iter =', max_iter)

    lr = linear_model.LogisticRegression(max_iter=max_iter)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Trigger a warning.
        lr.fit(x,y)

    print(type(w))
    print(w)
    try:
        warn = w[-1].category
        if warn is sklearn.exceptions.ConvergenceWarning:
            print('Convergence warning')

        #print(w[-1].message)

    except:
        print('No warning occurred')


