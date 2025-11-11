# attacks/boundary_attack.py
from __future__ import print_function
import sys, os
import click
import time
import numpy as np

# allow running from repo root like the other attack files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.output import disablePrint, enablePrint
except Exception:
    # fallback no-op
    def disablePrint(*a, **k): pass
    def enablePrint(*a, **k): pass


class BoundaryAttack(object):
    """
    Simple, pure-numpy Boundary (black-box) attack.
    - Python 2 compatible.
    - Per-sample (slow). It relies only on model.predict(X_batch) to get labels/probs.
    """

    def __init__(self, model, init_num_trials=500, max_iterations=1000,
                 delta=0.01, epsilon=1e-3, init_const=0.05, seed=None, verbose=False):
        self.model = model
        self.init_num_trials = int(init_num_trials)
        self.max_iterations = int(max_iterations)
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.init_const = float(init_const)
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose

    def _predict_labels(self, X):
        """
        Return class indices for X (batch). Works with outputs that are logits/probs or class indices.
        X: shape (N,H,W,C) or (1,H,W,C)
        """
        preds = self.model.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
            return preds.astype(int).reshape(-1)
        if preds.ndim == 2:
            return np.argmax(preds, axis=1)
        # fallback: flatten last dims then argmax
        return np.argmax(preds.reshape((preds.shape[0], -1)), axis=1)

    def _is_adversarial(self, cand, true_label):
        """
        cand: single sample (H,W,C) or batch
        true_label: scalar label
        """
        if cand.ndim == 3:
            lab = self._predict_labels(cand[np.newaxis, ...])[0]
        else:
            lab = self._predict_labels(cand)[0]
        return lab != int(true_label)

    def _l2(self, a, b):
        return np.linalg.norm((a - b).ravel())

    def _find_initial_adversarial(self, x_orig, y_true):
        """
        Random sampling to find an initial adversarial close-ish to domain
        """
        x0 = x_orig.astype(np.float32)
        for t in range(self.init_num_trials):
            noise = self.rng.rand(*x0.shape).astype(np.float32)
            alpha = self.rng.uniform(0.1, 0.9)
            cand = np.clip(alpha * noise + (1 - alpha) * x0, 0., 1.)
            if self._is_adversarial(cand, y_true):
                if self.verbose:
                    print("[BoundaryAttack] found initial adversarial at trial %d (alpha=%.3f)" % (t, alpha))
                return cand
        return None

    def generate_single(self, x_orig, y_true):
        """
        Generate adversarial for a single sample x_orig (H,W,C) and scalar y_true.
        Returns adv sample (H,W,C).
        """
        x_orig = x_orig.astype(np.float32)
        init_adv = self._find_initial_adversarial(x_orig, y_true)
        if init_adv is None:
            raise RuntimeError("BoundaryAttack: failed to find an initial adversarial after %d trials" % self.init_num_trials)

        current = init_adv.copy()
        orig = x_orig.copy()
        step_size = self.delta
        const = self.init_const

        for it in range(1, self.max_iterations + 1):
            # direction toward original
            direction = (orig - current)
            dir_flat = direction.ravel()
            dir_norm = np.linalg.norm(dir_flat)
            if dir_norm == 0:
                direction_unit = np.zeros_like(direction)
            else:
                direction_unit = direction / (dir_norm + 1e-12)

            current_dist = np.linalg.norm((current - orig).ravel())

            # step toward original
            candidate = current + step_size * direction_unit * current_dist

            # orthogonal perturbation
            orth = self.rng.randn(*candidate.shape).astype(np.float32)
            orth_flat = orth.ravel()
            if dir_norm == 0:
                orth_proj = orth_flat
            else:
                dot = np.dot(orth_flat, dir_flat) / (dir_norm**2 + 1e-12)
                proj = dot * dir_flat
                orth_proj = orth_flat - proj
            orth_proj = orth_proj.reshape(candidate.shape)
            orth_norm = np.linalg.norm(orth_proj.ravel())
            orth_unit = orth_proj / (orth_norm + 1e-12)

            candidate = candidate + const * orth_unit * current_dist
            candidate = np.clip(candidate, 0., 1.)

            if self._is_adversarial(candidate, y_true):
                current = candidate
                step_size = max(step_size * 0.99, 1e-12)
                const = max(const * 0.99, 1e-12)
            else:
                step_size *= 0.5
                const *= 0.5

            new_dist = np.linalg.norm((current - orig).ravel())
            if self.verbose and (it % 50 == 0 or it == 1):
                print("[BoundaryAttack] iter %d, dist %.6f, step %.6e, const %.6e" % (it, new_dist, step_size, const))

            if new_dist <= self.epsilon:
                if self.verbose:
                    print("[BoundaryAttack] distance below epsilon %.6e at iter %d" % (self.epsilon, it))
                break
            if step_size < 1e-12 or const < 1e-12:
                if self.verbose:
                    print("[BoundaryAttack] step/const too small; stopping")
                break

        return current


def generate_boundary_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Repo-compatible generator function.
    Signature matches other attack modules.
    Returns X_adv (numpy array).
    """
    accepted_params = ['batch_size', 'init_num_trials', 'max_iterations', 'delta', 'epsilon', 'init_const', 'seed', 'targeted']
    for k in attack_params:
        if k not in accepted_params:
            raise NotImplementedError("Unsupported params in Boundary attack: %s" % k)

    if 'targeted' in attack_params:
        del attack_params['targeted']

    batch_size = int(attack_params.get('batch_size', 1))
    init_num_trials = int(attack_params.get('init_num_trials', 500))
    max_iterations = int(attack_params.get('max_iterations', 1000))
    delta = float(attack_params.get('delta', 0.01))
    epsilon = float(attack_params.get('epsilon', 1e-3))
    init_const = float(attack_params.get('init_const', 0.05))
    seed = None
    if 'seed' in attack_params:
        seed = int(attack_params['seed'])

    N = len(X)
    X_adv_list = []

    if not verbose:
        disablePrint(attack_log_fpath)

    # progress bar similar style to Carlini files
    with click.progressbar(range(0, N), file=sys.stderr, show_pos=True,
                           width=40, bar_template='  [%(bar)s] Boundary Attacking %(info)s',
                           fill_char='>', empty_char='-') as bar:
        for i in bar:
            xi = X[i]
            # determine label (supports one-hot Y or label array)
            if Y is not None and hasattr(Y, 'shape') and getattr(Y, 'ndim', None) == 2:
                yi = int(np.argmax(Y[i]))
            elif Y is not None:
                yi = int(Y[i])
            else:
                # fallback: infer label from model
                yi = int(np.argmax(model.predict(xi[np.newaxis, ...])))

            attacker = BoundaryAttack(model,
                                      init_num_trials=init_num_trials,
                                      max_iterations=max_iterations,
                                      delta=delta,
                                      epsilon=epsilon,
                                      init_const=init_const,
                                      seed=(None if seed is None else (seed + i)),
                                      verbose=verbose)
            try:
                if not verbose:
                    disablePrint(attack_log_fpath)
                xi_adv = attacker.generate_single(xi, yi)
                if not verbose:
                    enablePrint()
            except Exception as e:
                # failure: keep original sample (consistent with earlier patterns)
                if verbose:
                    print("[BoundaryAttack] failed on sample %d: %s" % (i, str(e)))
                xi_adv = xi.copy()

            X_adv_list.append(xi_adv)

    if not verbose:
        enablePrint()

    X_adv = np.stack(X_adv_list, axis=0)
    return X_adv
