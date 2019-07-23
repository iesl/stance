"""
Code from https://github.com/gpeyre/SinkhornAutoDiff but changed to be batched 
"""

import torch
from torch.autograd import Variable


def sinkhorn_normalized(x, y, epsilon, n, niter):

    Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
    return 2 * Wxy - Wxx - Wyy


def batch_sinkhorn_loss(C, C_mask, epsilon=1, niter=100):
    """
    
    :param C: Batch size by MSL by MSL
    :param C_mask: Batch size by MSL by MSL 
    :param epsilon: 
    :param n: 
    :param niter: 
    :return: 
    """
    # B by MSL
    mu = C_mask[:,:,0]
    mu = mu / mu.sum(dim=1, keepdim=True)

    nu = C_mask[:,0,:]
    nu = nu / nu.sum(dim=1, keepdim=True)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A,dim):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(dim=dim, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN
    batch_size = C_mask.size(0)
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (
            torch.log(mu) # B by MSL
            - lse(M(u, v), # M = B by MSL by MSL, lse should sum along the columns
                  dim=2).squeeze()) \
            + u
        v = epsilon * (torch.log(nu) - lse(M(u, v),dim=1).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum() / batch_size

        actual_nits += 1
        if (err < thresh).data.cpu().numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    return pi

def sinkhorn_loss(C, epsilon, n, niter):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :

    # both marginals are fixed with equal weights
    mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion


    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.cpu().numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    return pi


if __name__ == "__main__":
    main()