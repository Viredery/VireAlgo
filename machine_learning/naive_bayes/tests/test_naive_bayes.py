# Author: viredery


import sys
sys.path.append("..")

import numpy as np

from naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

if __name__ == '__main__':
    X = np.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                  [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]]).T
    y = np.array( [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])

    nb = MultinomialNB(alpha=1.0,fit_prior=True)
    nb.fit(X,y)

    print(nb.predict(np.array([[2,4],[3,6],[2,5]])))

    nb = GaussianNB(alpha=0.0)
    print(nb.fit(X,y).predict(np.array([[2,4],[3,6],[2,5]])))

    X = np.array([[1,1,1,1,0,0,0],
                  [0,1,0,0,0,1,1]]).T
    y = np.array( [0,1,1,0,0,0,1])

    nb = BernoulliNB(alpha=0.0)
    print(nb.fit(X,y).predict(np.array([[0,0],[1,0],[1,1]])))
