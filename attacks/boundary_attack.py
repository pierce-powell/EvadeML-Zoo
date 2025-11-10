# attacks/boundary.py
from __future__ import print_function
import numpy as np

class BoundaryAttack(object):
    """
    Simple Boundary Attack implementation (Python 2 compatible)
    Based on: Brendel et al. (ICLR 2018)
    """

    def __init__(self, model, max_iter=5000, step_size=0.01, init_epsilon=0.1, targeted=False):
        self.model = model
        self.max_iter = max_iter
        self.step_size = step_size
        self.init_epsilon = init_epsilon
        self.targeted = targeted

    def perturb(self, x, y):
        """
        Perform Boundary Attack on input x with label y.
        """
        adv = self._find_initial_adversarial(x, y)
        if adv is None:
            print("[BoundaryAttack] Could not find initial adversarial example.")
            return x

        for i in xrange(self.max_iter):
            # random perturbation orthogonal to direction
            direction = adv - x
            direction /= np.linalg.norm(direction)

            perturb = np.random.randn(*x.shape)
            perturb /= np.linalg.norm(perturb)
            perturb = perturb - np.dot(perturb.flatten(), direction.flatten()) * direction
            perturb = perturb / np.linalg.norm(perturb)

            candidate = adv + self.step_size * perturb
            candidate = np.clip(candidate, 0, 1)

            # move closer if still adversarial
            if self._is_adversarial(candidate, y):
                adv = candidate
            else:
                # small move toward original
                adv = adv + (self.init_epsilon * (x - adv))
                adv = np.clip(adv, 0, 1)

            if i % 50 == 0:
                print("[BoundaryAttack] iter %d / %d" % (i, self.max_iter))

        return adv

    def _find_initial_adversarial(self, x, y):
        """
        Try to find an initial adversarial sample by adding random noise.
        """
        for _ in xrange(1000):
            noise = np.random.uniform(0, 1, size=x.shape)
            candidate = np.clip(noise, 0, 1)
            if self._is_adversarial(candidate, y):
                return candidate
        return None

    def _is_adversarial(self, x, y):
        pred = np.argmax(self.model.predict(np.array([x])), axis=1)[0]
        return pred != y if not self.targeted else pred == y
