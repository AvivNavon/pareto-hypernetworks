import logging
from abc import abstractmethod
from solvers.epo import EPO  # We use the same solver as EPO
from torch.autograd import Variable

import numpy as np
import torch

"""Implementation of:
1. Linear scalarization
2. PMTL
3. EPO

The EPO and PMTL implementations are modifications of the code in https://github.com/dbmptr/EPOSearch 

"""


class Baseline:
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

    @abstractmethod
    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        pass

    def __call__(self, losses, ray, parameters, **kwargs):
        return self.get_weighted_loss(losses, ray, parameters, **kwargs)


class LinearScalarizationBaseline(Baseline):

    def __init__(self, n_tasks):
        super().__init__(n_tasks)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        return (losses * ray).sum()


class EPOBaseline(Baseline):

    def __init__(self, n_tasks, n_params):
        super().__init__(n_tasks)
        self.solver = EPO(n_tasks=n_tasks, n_params=n_params)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        assert parameters is not None
        return self.solver.get_weighted_loss(losses, ray, parameters)


class PMTLBaseline(Baseline):

    def __init__(self, n_tasks, rays, max_init_steps=2):
        super().__init__(n_tasks)
        self.rays = rays
        self.iter = 0
        self.max_init_steps = max_init_steps
        self.init_end = False

    def reset_init(self):
        self.iter = 0
        self.init_end = False

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        grads = []
        for i, loss in enumerate(losses):
            loss = losses[i]
            g = torch.autograd.grad(loss, parameters, retain_graph=True)
            flat_grad = self._flattening(g)
            grads.append(flat_grad.data)

        grads = torch.stack(grads)

        if self.init_end:
            weights = self.get_d_paretomtl(grads, losses, ray)
            # NOTE: the x self.n_tasks is from the PMTL code and does not appear in the EPO code
            normalized_const = self.n_tasks / torch.sum(torch.abs(weights))
            weights *= normalized_const

        else:
            flag, weights = self.get_d_paretomtl_init(grads, losses, ray)
            if flag:
                logging.info(f"PMTL finished step 1 at iter {self.iter}")
                self.init_end = True

        self.iter += 1
        if self.iter >= self.max_init_steps:
            self.init_end = True

        weights = weights.to(losses.device)
        weighted_loss = torch.sum(losses * weights)

        return weighted_loss

    def get_d_paretomtl_init(self, grads, losses, ray):
        """
        calculate the gradient direction for ParetoMTL initialization
        """

        flag = False
        nobj = losses.shape

        # check active constraints
        current_weight = ray
        rest_weights = self.rays
        w = rest_weights - current_weight

        gx = torch.matmul(w, losses / torch.norm(losses))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            flag = True
            return flag, torch.zeros(nobj)
        if torch.sum(idx) == 1:
            sol = torch.ones(1).float()
        else:
            vec = torch.matmul(w[idx], grads)
            sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        new_weights = []
        for t in range(len(losses)):
            new_weights.append(torch.sum(torch.stack([sol[j] * w[idx][j, t] for j in torch.arange(0, torch.sum(idx))])))

        return flag, torch.stack(new_weights)

    def get_d_paretomtl(self, grads, losses, ray):
        """Calculate the gradient direction for ParetoMTL"""

        # check active constraints
        current_weight = ray
        rest_weights = self.rays
        w = rest_weights - current_weight

        gx = torch.matmul(w, losses / torch.norm(losses))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.tensor(sol).float().to(self.rays.device)

        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        new_weights = []
        for t in range(len(losses)):
            new_weights.append(
                sol[t] + torch.sum(
                    torch.stack([sol[j] * w[idx][j - 2, t] for j in torch.arange(2, 2 + torch.sum(idx))]))
            )

        return torch.stack(new_weights)


class MGDABaseline(Baseline):
    """Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.solver = MinNormSolver()

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_weighted_loss(self, losses, ray, shared_parameters=None, return_sol=False, **kwargs):
        """Note that we do not use ray here!!!

        :param losses:
        :param ray:
        :param shared_parameters:
        :param return_sol:
        :param kwargs:
        :return:
        """
        # grads = []
        # for i, loss in enumerate(losses):
        #     g = torch.autograd.grad(loss, shared_parameters, retain_graph=True)
        #     flat_grad = self._flattening(g)
        #     grads.append(flat_grad.data)
        #
        # grads = torch.stack(grads)
        #
        # sol, min_norm = MinNormSolver.find_min_norm_element([grads[[t]] for t in range(len(grads))])

        # NOTE: original code
        grads = {}
        for i, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            grads[i] = []
            for param in shared_parameters:
                if param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                    # NOTE: need to free grads...
                    param.grad = None

        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(len(grads))])

        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        if return_sol:
            return weighted_loss, sol, min_norm

        return weighted_loss


# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018
# https://github.com/intel-isl/MultiObjectiveOptimization
class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k],
                                                 vecs[j][k]).item()  # torch.dot(vecs[i][k], vecs[j][k]).data[0]
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k],
                                                 vecs[i][k]).item()  # torch.dot(vecs[i][k], vecs[i][k]).data[0]
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k],
                                                 vecs[j][k]).item()  # torch.dot(vecs[j][k], vecs[j][k]).data[0]
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


class MGDABaseline(Baseline):
    """Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization
    """
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.solver = MinNormSolver()

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_weighted_loss(self, losses, ray, shared_parameters=None, return_sol=False, **kwargs):
        """Note that we do not use ray here!!!
        :param losses:
        :param ray:
        :param shared_parameters:
        :param return_sol:
        :param kwargs:
        :return:
        """
        grads = []
        for i, loss in enumerate(losses):
            g = torch.autograd.grad(loss, shared_parameters, retain_graph=True)
            flat_grad = self._flattening(g)
            grads.append(flat_grad.data)

        grads = torch.stack(grads)

        sol, min_norm = MinNormSolver.find_min_norm_element([grads[[t]] for t in range(len(grads))])
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        if return_sol:
            return weighted_loss, sol, min_norm

        return weighted_loss


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn
